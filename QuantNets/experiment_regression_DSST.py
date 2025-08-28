# DSST-specific regression experiment class
import shutil
import os
import sys
import torch
import pickle
import torch.nn.functional as F
import torch
import torch_geometric
import time
import statistics
import matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader as RawDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader

from util.data_processing import *


# define a wrapper time_it decorator function
def time_it(func):
    def wrapper_function(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        stop = time.time()
        print(f'Function {func.__name__} took: {stop-start}s')
        return res
    return wrapper_function


class ExperimentRegression:
    # list of static variables ...
    experiment_id = 1 # static variable for creating experiments dir ...

    def __init__(self, 
                 sgcn_model = None, 
                 qgcn_model = None, 
                 cnn_model = None,
                 optim_params = None, 
                 base_path = ".", 
                 num_train = None,
                 num_test = None,
                 dataset_name = None,
                 train_batch_size = 64,
                 test_batch_size = 64,
                 train_shuffle_data = True,
                 test_shuffle_data = False,
                 profile_run = False,
                 walk_clock_num_runs = 10,
                 id = None):
        
        # Controls whether we want to print runtime per model
        self.profile_run = profile_run
        self.walk_clock_num_runs = walk_clock_num_runs

        # Load the DSST dataset from cached graph files
        data_struct = read_cached_graph_dataset(
            num_train=num_train, 
            num_test=num_test, 
            dataset_name=dataset_name, 
            parent_dir=base_path
        )

        # Save the references to the datasets
        self.data_struct = data_struct
        
        # For DSST brain connectivity graphs, we don't have raw image data
        # So we'll create empty datasets for CNN compatibility
        raw_train_data = []
        raw_test_data = []

        # Get the geometric graph data for brain connectivity
        geometric_qgcn_train_data = data_struct["geometric"].get("gcn_train_data", None)
        geometric_qgcn_test_data  = data_struct["geometric"].get("gcn_test_data", None)
        geometric_qgcn_train_data = geometric_qgcn_train_data or data_struct["geometric"].get("qgcn_train_data", None)
        geometric_qgcn_test_data  = geometric_qgcn_test_data or data_struct["geometric"].get("qgcn_test_data", None)
        geometric_sgcn_train_data = data_struct["geometric"]["sgcn_train_data"]
        geometric_sgcn_test_data  = data_struct["geometric"]["sgcn_test_data"]

        print('DSST Dataset Info:')
        print(f'Geometric qgcn dataset train: {len(geometric_qgcn_train_data)}')
        print(f'Geometric qgcn dataset test: {len(geometric_qgcn_test_data)}')
        print(f'Geometric sgcn dataset train: {len(geometric_sgcn_train_data)}')
        print(f'Geometric sgcn dataset test: {len(geometric_sgcn_test_data)}')
        
        # Print dataset statistics
        if geometric_sgcn_train_data:
            sample_graph = geometric_sgcn_train_data[0]
            print(f'Sample graph - Nodes: {sample_graph.num_nodes}, Edges: {sample_graph.num_edges}')
            print(f'Node features shape: {sample_graph.x.shape}')
            print(f'Edge features shape: {sample_graph.edge_attr.shape if hasattr(sample_graph, "edge_attr") else "None"}')
            print(f'Target (DSST score): {sample_graph.y.item()}')

        # Create empty raw data loaders (not used for brain connectivity)
        raw_train_loader = RawDataLoader([], batch_size=train_batch_size, shuffle=False)
        raw_test_loader = RawDataLoader([], batch_size=test_batch_size, shuffle=False)

        # Create graph data loaders for brain connectivity
        shuffle_qgcn_geo_train_data = (len(geometric_qgcn_train_data) != 0) and train_shuffle_data
        geometric_qgcn_train_loader = GraphDataLoader(
            geometric_qgcn_train_data, 
            batch_size=train_batch_size, 
            shuffle=shuffle_qgcn_geo_train_data
        )
        shuffle_qgcn_geo_test_data = (len(geometric_qgcn_test_data) != 0) and test_shuffle_data
        geometric_qgcn_test_loader = GraphDataLoader(
            geometric_qgcn_test_data,
            batch_size=test_batch_size,
            shuffle=shuffle_qgcn_geo_test_data
        )
        shuffle_sgcn_geo_train_data = (len(geometric_sgcn_train_data) != 0) and train_shuffle_data
        geometric_sgcn_train_loader = GraphDataLoader(
            geometric_sgcn_train_data, 
            batch_size=train_batch_size, 
            shuffle=shuffle_sgcn_geo_train_data
        )
        shuffle_sgcn_geo_test_data = (len(geometric_sgcn_test_data) != 0) and test_shuffle_data
        geometric_sgcn_test_loader = GraphDataLoader(
            geometric_sgcn_test_data,
            batch_size=test_batch_size,
            shuffle=shuffle_sgcn_geo_test_data
        )

        # Save the references here
        self.raw_train_dataloader = raw_train_loader
        self.raw_test_dataloader = raw_test_loader
        self.sp_qgcn_train_dataloader = geometric_qgcn_train_loader
        self.sp_qgcn_test_dataloader = geometric_qgcn_test_loader
        self.sp_sgcn_train_dataloader = geometric_sgcn_train_loader
        self.sp_sgcn_test_dataloader = geometric_sgcn_test_loader

        # Define what device we are using
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.deterministic = True
        print(f'Using device: {self.device}')

        # Define the experiments folder and add the directory for this run
        self.cache_run = True
        self.specific_run_dir = None
        self.cnn_specific_run_dir = None
        self.qgcn_specific_run_dir = None
        self.sgcn_specific_run_dir = None
        
        if base_path == None:
            self.cache_run = False
        else:
            local_experiment_id = ExperimentRegression.experiment_id
            if id == None:
                ExperimentRegression.experiment_id += 1
            else:
                local_experiment_id = id
            self.local_experiment_id = local_experiment_id
            # Create the folder structure for this run
            self.__create_experiment_folder_structure(base_path, local_experiment_id)

        # Save the models
        self.cnn_model = cnn_model  # Will be None for DSST
        self.qgcn_model = qgcn_model
        self.sgcn_model = sgcn_model

        # Assert that at least one graph model exists for DSST
        assert any([self.qgcn_model, self.sgcn_model]), "At least one graph model (QGCN or SGCN) must be provided for DSST brain connectivity analysis"
        self.cnn_model_exists = cnn_model != None
        self.qgcn_model_exists = qgcn_model != None
        self.sgcn_model_exists = sgcn_model != None

        # Put the models on the device
        if self.cnn_model_exists:
            self.cnn_model.to(self.device)
        if self.qgcn_model_exists:
            self.qgcn_model.to(self.device)
        if self.sgcn_model_exists:
            self.sgcn_model.to(self.device)

        # Define the optimizers for the different models
        self.optim_params = optim_params
        learning_rate = 0.001  # Default learning rate for brain connectivity regression
        if optim_params != None and "lr" in optim_params.keys():
            learning_rate = optim_params["lr"] 
        
        self.cnn_model_optimizer = None
        self.qgcn_model_optimizer = None
        self.sgcn_model_optimizer = None
        
        if self.cnn_model_exists:
            self.cnn_model_optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=learning_rate)
        if self.qgcn_model_exists:
            self.qgcn_model_optimizer = torch.optim.Adam(self.qgcn_model.parameters(), lr=learning_rate)
        if self.sgcn_model_exists:
            self.sgcn_model_optimizer = torch.optim.Adam(self.sgcn_model.parameters(), lr=learning_rate)

        # Print model statistics if profiling
        if self.profile_run: 
            self.__print_models_stats()

    def __print_models_stats(self):
        """Print model statistics for DSST brain connectivity models."""
        try:
            from flops_counter.ptflops import get_model_complexity_info
        except ImportError:
            print("ptflops not available for model profiling")
            return

        if self.sgcn_model_exists:
            # Pick the largest data (by node degree) to profile all the models
            crit_lst = [data.x.numel() + data.edge_index.numel() for data in self.data_struct["geometric"]["sgcn_train_data"]]
            _, max_crit_index = max(crit_lst), crit_lst.index(max(crit_lst))
            data_sample = self.data_struct["geometric"]["sgcn_train_data"][max_crit_index].clone().detach().to(self.device)
            model = self.sgcn_model
            model.eval()
            
            try:
                macs, params = get_model_complexity_info(model, data_sample, as_strings=False, print_per_layer_stat=False, verbose=False)
                flops, macs, params = round(2*macs / 1e3, 3), round(macs / 1e3, 3), round(params / 1e3, 3)
                
                # Profile Inference Wall Time
                wall_times = []
                for _ in range(self.walk_clock_num_runs):
                    start_time = time.time()
                    _ = model(self.data_struct["geometric"]["sgcn_train_data"][max_crit_index].clone().detach().to(self.device))
                    end_time = time.time()
                    wall_times.append(end_time - start_time)
                wall_time_mean = statistics.mean(wall_times)
                wall_time_std = statistics.stdev(wall_times)
                
                print("\n-----------------")
                print("SGCN Model Stats (DSST Brain Connectivity):")
                print(f"Profiling data sample: {self.data_struct['geometric']['sgcn_train_data'][max_crit_index]}")
                print("-------------------------------------------------------------------------------------------")
                print(f'Number of parameters: {params} k')
                print(f'Theoretical Computational Complexity (FLOPs): {flops} kFLOPs')
                print(f'Theoretical Computational Complexity (MACs):  {macs} kMACs')
                print(f'Wall Clock  Computational Complexity (s):     {wall_time_mean} +/- {wall_time_std} s ')
                print("-------------------------------------------------------------------------------------------")
            except Exception as e:
                print(f"Error profiling SGCN model: {e}")

        if self.qgcn_model_exists:
            # Similar profiling for QGCN
            crit_lst = [data.x.numel() + data.edge_index.numel() for data in self.data_struct["geometric"]["qgcn_train_data"]]
            _, max_crit_index = max(crit_lst), crit_lst.index(max(crit_lst))
            data_sample = self.data_struct["geometric"]["qgcn_train_data"][max_crit_index].clone().detach().to(self.device)
            model = self.qgcn_model
            model.eval()
            
            try:
                macs, params = get_model_complexity_info(model, data_sample, as_strings=False, print_per_layer_stat=False, verbose=False)
                flops, macs, params = round(2*macs / 1e3, 3), round(macs / 1e3, 3), round(params / 1e3, 3)
                
                # Profile Inference Wall Time
                wall_times = []
                for _ in range(self.walk_clock_num_runs):
                    start_time = time.time()
                    _ = model(self.data_struct["geometric"]["qgcn_train_data"][max_crit_index].clone().detach().to(self.device))
                    end_time = time.time()
                    wall_times.append(end_time - start_time)
                wall_time_mean = statistics.mean(wall_times)
                wall_time_std = statistics.stdev(wall_times)
                
                print("\n-----------------")
                print("QGCN Model Stats (DSST Brain Connectivity):")
                print(f"Profiling data sample: {self.data_struct['geometric']['qgcn_train_data'][max_crit_index]}")
                print("-------------------------------------------------------------------------------------------")
                print(f'Number of parameters: {params} k')
                print(f'Theoretical Computational Complexity (FLOPs): {flops} kFLOPs')
                print(f'Theoretical Computational Complexity (MACs):  {macs} kMACs')
                print(f'Wall Clock  Computational Complexity (s):     {wall_time_mean} +/- {wall_time_std} s ')
                print("-------------------------------------------------------------------------------------------")
            except Exception as e:
                print(f"Error profiling QGCN model: {e}")

    def __create_experiment_folder_structure(self, base_path, experiment_id):
        """Create folder structure for experiment results."""
        if not os.path.exists(base_path):
            print("Ensure that your base path exists -> {}".format(base_path))
            sys.exit(1)
        experiments_dir = os.path.join(base_path, "Experiments_DSST")
        if not os.path.exists(experiments_dir):
            os.mkdir(experiments_dir)
        underscored_experiment_id = "_".join(str(experiment_id).strip().split(" "))
        specific_run_dir = os.path.join(experiments_dir, "run_" + underscored_experiment_id)
        if not os.path.exists(specific_run_dir):
            os.mkdir(specific_run_dir)
        self.specific_run_dir = specific_run_dir

        # Create the respective folders for this run
        qgcn_specific_run_dir = os.path.join(specific_run_dir, "qgcn")
        if not os.path.exists(qgcn_specific_run_dir):
            os.mkdir(qgcn_specific_run_dir)
        self.qgcn_specific_run_dir = qgcn_specific_run_dir
        
        sgcn_specific_run_dir = os.path.join(specific_run_dir, "sgcn")
        if not os.path.exists(sgcn_specific_run_dir):
            os.mkdir(sgcn_specific_run_dir)
        self.sgcn_specific_run_dir = sgcn_specific_run_dir

        # Copy architecture files (assuming they exist)
        try:
            gnn_source_filepath = os.path.join(base_path, "gnn", "architectures.py")
            if os.path.exists(gnn_source_filepath):
                qgcn_destination_filepath = os.path.join(self.qgcn_specific_run_dir, "architectures.py")
                if not os.path.exists(qgcn_destination_filepath):
                    shutil.copyfile(gnn_source_filepath, qgcn_destination_filepath)
                sgcn_destination_filepath = os.path.join(self.sgcn_specific_run_dir, "architectures.py")
                if not os.path.exists(sgcn_destination_filepath):
                    shutil.copyfile(gnn_source_filepath, sgcn_destination_filepath)
        except Exception as e:
            print(f"Warning: Could not copy architecture files: {e}")

    def __cache_models(self):
        """Save trained models to disk."""
        if self.qgcn_specific_run_dir != None and self.qgcn_model_exists:
            qgcn_model_filepath = os.path.join(self.qgcn_specific_run_dir, "model.pth")
            torch.save(self.qgcn_model, qgcn_model_filepath)
        if self.sgcn_specific_run_dir != None and self.sgcn_model_exists:
            sgcn_model_filepath = os.path.join(self.sgcn_specific_run_dir, "model.pth")
            torch.save(self.sgcn_model, sgcn_model_filepath)

    def __cache_results(self, train_qgcn_loss_array, train_sgcn_loss_array, 
                        train_qgcn_mse_array, train_sgcn_mse_array,
                        test_qgcn_mse_array, test_sgcn_mse_array):
        """Save training results to disk."""
        if self.qgcn_specific_run_dir != None and self.qgcn_model_exists:
            train_qgcn_loss_filepath = os.path.join(self.qgcn_specific_run_dir, "train_loss.pk")
            train_qgcn_mse_filepath = os.path.join(self.qgcn_specific_run_dir, "train_mse.pk")
            test_qgcn_mse_filepath = os.path.join(self.qgcn_specific_run_dir, "test_mse.pk")
            with open(train_qgcn_loss_filepath, 'wb') as f:
                pickle.dump(train_qgcn_loss_array, f)
            with open(train_qgcn_mse_filepath, 'wb') as f:
                pickle.dump(train_qgcn_mse_array, f)
            with open(test_qgcn_mse_filepath, 'wb') as f:
                pickle.dump(test_qgcn_mse_array, f)
        
        if self.sgcn_specific_run_dir != None and self.sgcn_model_exists:
            train_sgcn_loss_filepath = os.path.join(self.sgcn_specific_run_dir, "train_loss.pk")
            train_sgcn_mse_filepath = os.path.join(self.sgcn_specific_run_dir, "train_mse.pk")
            test_sgcn_mse_filepath = os.path.join(self.sgcn_specific_run_dir, "test_mse.pk")
            with open(train_sgcn_loss_filepath, 'wb') as f:
                pickle.dump(train_sgcn_loss_array, f)
            with open(train_sgcn_mse_filepath, 'wb') as f:
                pickle.dump(train_sgcn_mse_array, f)
            with open(test_sgcn_mse_filepath, 'wb') as f:
                pickle.dump(test_sgcn_mse_array, f)

    def __move_graph_data_to_device(self, data):
        """Move graph data to target device."""
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x.to(self.device)
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            data.edge_index = data.edge_index.to(self.device)
        if hasattr(data, 'y') and data.y is not None:
            data.y = data.y.to(self.device)
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = data.pos.to(self.device)
        if hasattr(data, 'batch') and data.batch is not None:
            data.batch = data.batch.to(self.device)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.to(self.device)
        return data

    def __train(self):
        """Train models for one epoch."""
        # For QGCN training
        qgcn_loss_all, qgcn_total_graphs = 0, 0
        if self.qgcn_model_exists:
            self.qgcn_model.train()
            if self.profile_run: start_time = time.time()
            for data in self.sp_qgcn_train_dataloader:
                qgcn_data = self.__move_graph_data_to_device(data)
                qgcn_total_graphs += qgcn_data.num_graphs
                
                self.qgcn_model_optimizer.zero_grad()
                qgcn_output = self.qgcn_model(qgcn_data)
                qgcn_loss = F.mse_loss(qgcn_output, qgcn_data.y.float())
                qgcn_loss.backward()
                qgcn_loss_all += qgcn_data.num_graphs * qgcn_loss.item()
                self.qgcn_model_optimizer.step()
            if self.profile_run: 
                stop_time = time.time()
                profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
                print(f"Single epoch training... qgcn_model done{profile_stats}")

        # For SGCN training
        sgcn_loss_all, sgcn_total_graphs = 0, 0
        if self.sgcn_model_exists:
            self.sgcn_model.train()
            if self.profile_run: start_time = time.time()
            for data in self.sp_sgcn_train_dataloader:
                sgcn_data = self.__move_graph_data_to_device(data)
                sgcn_total_graphs += sgcn_data.num_graphs
                
                self.sgcn_model_optimizer.zero_grad()
                sgcn_output = self.sgcn_model(sgcn_data)
                sgcn_loss = F.mse_loss(sgcn_output, sgcn_data.y.float())
                sgcn_loss.backward()
                sgcn_loss_all += sgcn_data.num_graphs * sgcn_loss.item()
                self.sgcn_model_optimizer.step()
            if self.profile_run: 
                stop_time = time.time()
                profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
                print(f"Single epoch training... sgcn_model done{profile_stats}")

        # Cache models if needed
        if self.cache_run:
            self.__cache_models()

        # Normalize loss by the total length of training set
        if self.qgcn_model_exists:
            qgcn_loss_all /= qgcn_total_graphs
        if self.sgcn_model_exists:
            sgcn_loss_all /= sgcn_total_graphs

        return qgcn_loss_all, sgcn_loss_all

    def __evaluate(self, eval_train_data=False):
        """Evaluate models on train or test data."""
        # For QGCN evaluation
        qgcn_mse = 0
        if self.qgcn_model_exists:
            sp_qgcn_dataset_loader = self.sp_qgcn_train_dataloader if eval_train_data else self.sp_qgcn_test_dataloader
            self.qgcn_model.eval()
            if self.profile_run: start_time = time.time()
            total_mse, total_samples = 0, 0
            with torch.no_grad():
                for data in sp_qgcn_dataset_loader:
                    qgcn_data = self.__move_graph_data_to_device(data)
                    pred = self.qgcn_model(qgcn_data)
                    mse = F.mse_loss(pred, qgcn_data.y.float(), reduction='sum')
                    total_mse += mse.item()
                    total_samples += qgcn_data.num_graphs
            qgcn_mse = total_mse / total_samples
            if self.profile_run: 
                stop_time = time.time()
                profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
                print(f"{'train' if eval_train_data else 'test'} data: qgcn eval done{profile_stats}")

        # For SGCN evaluation
        sgcn_mse = 0
        if self.sgcn_model_exists:
            sp_sgcn_dataset_loader = self.sp_sgcn_train_dataloader if eval_train_data else self.sp_sgcn_test_dataloader
            self.sgcn_model.eval()
            if self.profile_run: start_time = time.time()
            total_mse, total_samples = 0, 0
            with torch.no_grad():
                for data in sp_sgcn_dataset_loader:
                    sgcn_data = self.__move_graph_data_to_device(data)
                    pred = self.sgcn_model(sgcn_data)
                    mse = F.mse_loss(pred, sgcn_data.y.float(), reduction='sum')
                    total_mse += mse.item()
                    total_samples += sgcn_data.num_graphs
            sgcn_mse = total_mse / total_samples
            if self.profile_run: 
                stop_time = time.time()
                profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
                print(f"{'train' if eval_train_data else 'test'} data: sgcn eval done{profile_stats}")

        return qgcn_mse, sgcn_mse

    @time_it
    def run(self, num_epochs=None, eval_training_set=True):
        """Run the complete training and evaluation loop."""
        if num_epochs == None or num_epochs <= 0:
            print("num_epochs ({}) in [ExperimentRegression.run] is invalid".format(num_epochs))
            sys.exit(1)

        # Define variables to hold the stats
        test_qgcn_mse_array, test_sgcn_mse_array = [], []
        train_qgcn_mse_array, train_sgcn_mse_array = [], []
        train_qgcn_loss_array, train_sgcn_loss_array = [], []
        
        print("Starting DSST Brain Connectivity Regression Training...")
        print(f"Training for {num_epochs} epochs")
        print(f"Models: QGCN={self.qgcn_model_exists}, SGCN={self.sgcn_model_exists}")
        
        for epoch in range(1, num_epochs + 1):
            # Time epoch operations
            start_time = time.time()
            print("training... epoch {}".format(epoch))
            
            qgcn_loss, sgcn_loss = self.__train()
            train_qgcn_loss_array.append(qgcn_loss)
            train_sgcn_loss_array.append(sgcn_loss)
            
            train_qgcn_mse, train_sgcn_mse = 0, 0
            if eval_training_set:
                train_qgcn_mse, train_sgcn_mse = self.__evaluate(eval_train_data=True)
            train_qgcn_mse_array.append(train_qgcn_mse)
            train_sgcn_mse_array.append(train_sgcn_mse)
            
            test_qgcn_mse, test_sgcn_mse = self.__evaluate(eval_train_data=False)
            test_qgcn_mse_array.append(test_qgcn_mse)
            test_sgcn_mse_array.append(test_sgcn_mse)
            
            stop_time = time.time()

            # Build the display string
            epoch_str = "Epoch: {:03d}, ".format(epoch)
            loss_str = "QGCN_Loss: {:.5f}, SGCN_Loss: {:.5f}, ".format(qgcn_loss, sgcn_loss)
            train_mse_str = "QGCN_Train_MSE: {:.5f}, SGCN_Train_MSE: {:.5f}, ".format(train_qgcn_mse, train_sgcn_mse)
            test_mse_str = "QGCN_Test_MSE: {:.5f}, SGCN_Test_MSE: {:.5f}, ".format(test_qgcn_mse, test_sgcn_mse)
            
            # Print out the results
            print("{}".format("".join([epoch_str, loss_str, train_mse_str, test_mse_str])))
            print(f"Epoch took a total of {stop_time - start_time}s")

        # Cache results if we need to
        if self.cache_run:
            self.__cache_results(
                train_qgcn_loss_array, train_sgcn_loss_array,
                train_qgcn_mse_array, train_sgcn_mse_array,
                test_qgcn_mse_array, test_sgcn_mse_array
            )
            
        print("DSST Brain Connectivity Regression Training Complete!")
        return {
            'train_qgcn_loss': train_qgcn_loss_array,
            'train_sgcn_loss': train_sgcn_loss_array,
            'train_qgcn_mse': train_qgcn_mse_array,
            'train_sgcn_mse': train_sgcn_mse_array,
            'test_qgcn_mse': test_qgcn_mse_array,
            'test_sgcn_mse': test_sgcn_mse_array
        }

    # Static method for plotting results
    @staticmethod
    def plot_history(data, labels):
        """Plot training history."""
        font = {'weight':'bold', 'size':8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams.update({'font.size':8})

        indicators = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        leftover_count = len(data) % 2
        num_rows = len(data) // 2
        col_count = 2
        row_count = num_rows + leftover_count
        fig, ax = plt.subplots(nrows=row_count, ncols=col_count, figsize=(12, 8))
        fig.tight_layout(pad=0.8)
        plt.subplots_adjust(wspace=0.4, hspace=0.5)

        # Loop and plot the data and their labels
        for i, row_plts in enumerate(ax):
            for j, row_col_plt in enumerate(row_plts):
                data_index = i * col_count + j
                if data_index < len(data):
                    xdata = list(range(1, len(data[data_index]) + 1))
                    ydata = data[data_index]
                    data_label = labels[data_index]
                    data_indicator = indicators[data_index % len(indicators)]
                    row_col_plt.plot(xdata, ydata, color=data_indicator, label=data_label)
                    row_col_plt.set_xticks(xdata)
                    row_col_plt.legend(loc="upper right")
                    row_col_plt.set_xlabel('Epoch')
                    row_col_plt.set_ylabel(data_label)
                    row_col_plt.set_title('{} vs. No. of epochs'.format(data_label))
                else:
                    row_col_plt.set_visible(False)
        plt.show()