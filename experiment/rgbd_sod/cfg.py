import os, torch
from typing import List, Tuple

from .run_type import RUN_TYPE, run_type

from .utils import device


class base_cfg:
    def __init__(
        self,
        epoch: int,
        experiment_name: str,
        datasets_set: int,
        run_type: RUN_TYPE,
    ):
        self.experiment_name = experiment_name
        self.datasets_set = datasets_set
        self.run_type = run_type
        if run_type == RUN_TYPE.KAGGLE:
            self.base_datasets_working_dir_path = "/kaggle/input"  # Colab

            self.datasets_working_dir_path = os.path.join(
                self.base_datasets_working_dir_path, f"rgbdsod-set{datasets_set}"
            )

            """Source code"""
            self.source_code_dir = os.path.join("/", "kaggle", "working", "sources")

            """Benchmark"""
            self.sotas_working_dir = "/home/sotas"
        else:
            # Default: same as RUN_TYPE.COLAB
            self.mount_path = "/content/drive"  # GoogleDrive
            self.datasets_dir_path = os.path.join(
                self.mount_path, "MyDrive", "RGBD_SOD", "datasets"
            )  # GoogleDrive
            self.base_datasets_working_dir_path = "/content/datasets"  # Colab
            self.datasets_working_dir_path = os.path.join(
                self.base_datasets_working_dir_path, f"v{datasets_set}"
            )  # Colab

            """Source code"""
            self.source_code_dir = os.path.join(
                self.mount_path, "MyDrive", "RGBD_SOD", "sources"
            )

            """Benchmark"""
            self.sotas_working_dir = "/content/sotas"

        if self.datasets_set == 1:
            """Set 1: COME15K
            Train dataset contains 8,025 image pairs of RGB-D
            We split our testing     dataset to a moderate-level testing set (“Normal”) and a
            hard testing set (“Difficult”) with 4,600 image pairs and
            3,000 pairs respectively"""
            self.test_dataset_names = [
                "COME-E",
                "COME-H",
                "DES",
                "DUT-RGBD",
                "LFSD",
                "NJU2K",
                "NLPR",
                "ReDWeb-S",
                "SIP",
                "STERE",
            ]

            # GoogleDrive
            self.datasets_dir = os.path.join(self.datasets_dir_path, "DatasetsV1")
            self.train_dataset_zip_path = os.path.join(self.datasets_dir, "train.zip")
            self.test_datasets_dir_path = os.path.join(self.datasets_dir, "test")
            self.dev_dataset_zip_path: str = None
            self.benchmark_dir_path = os.path.join(self.datasets_dir, "benchmark")

            # Colab
            self.train_dataset_working_dir_path = os.path.join(
                self.datasets_working_dir_path, "train"
            )
            self.test_datasets_working_dir_path = os.path.join(
                self.datasets_working_dir_path, "test"
            )
            self.dev_dataset_working_dir_path = os.path.join(
                self.datasets_working_dir_path, "test", "COME-E"
            )
        elif self.datasets_set == 2:
            """Set 2: Previous datasets"""
            self.test_dataset_names = [
                "DES",
                "LFSD",
                "NJU2K",
                "NLPR",
                "SIP",
                "SSD",
                "STERE",
            ]

            # GoogleDrive
            self.datasets_dir = os.path.join(self.datasets_dir_path, "DatasetsV2")
            self.train_dataset_zip_path = os.path.join(self.datasets_dir, "train.zip")
            self.test_datasets_dir_path = os.path.join(self.datasets_dir, "test")
            self.dev_dataset_zip_path: str = os.path.join(self.datasets_dir, "dev.zip")
            self.benchmark_dir_path = os.path.join(self.datasets_dir, "benchmark")

            # Colab
            self.train_dataset_working_dir_path = os.path.join(
                self.datasets_working_dir_path, "train"
            )
            self.test_datasets_working_dir_path = os.path.join(
                self.datasets_working_dir_path, "test"
            )
            self.dev_dataset_working_dir_path = os.path.join(
                self.datasets_working_dir_path, "dev"
            )
        else:
            raise NotImplementedError()

        """Evaluation benchmark"""
        self.benchmark_csv_dir_path = os.path.join(self.source_code_dir, "csv")
        self.evaluation_metrics_columns = [
            "Index",
            "Image path",
            "MAE",
            "S-measure",
            "F-measure (Max-F)",
            "E-measure (Max-E)",
        ]
        self.detail_evaluation_metrics_columns = [
            "Index",
            "Image path",
            "RGB",
            "Depth",
            "GT",
            "Prediction",
            "MAE",
            "S-measure",
            "F-measure (Max-F)",
            "E-measure (Max-E)",
        ]

        """Deployment"""
        self.deployment_dir_path = os.path.join(self.source_code_dir, "deployment")

        # -------------------------------------------------------------------------

        """Loggers"""
        self.logs_dir = os.path.join(self.source_code_dir, "logs")

        """Experiment"""
        self.experiment_dir_path = os.path.join(self.source_code_dir, "experiment")

        """Pickle"""
        self.pickle_dir_path = os.path.join(self.source_code_dir, "pickle")

        self.num_train_imgs: int
        self.niters_per_epoch: int
        self.test_image_size: int = 224
        self.image_size: int = 224

        """Gradient Accumulation"""
        self.accum_iter = 1

        """Wandb tracking"""
        self.wandb_api_key = (
            "c3fc6b778d58b02a1519dec88b08f0dae1fd5b3b"  # thinh.huynh.re@gmail.com
        )

        """Whether using fp16 instead of fp32 (default)"""
        self.is_fp16: bool = True

        """For hard reset learning rate"""
        self.is_hard_set_lr = False

        self.is_padding: bool = (
            False  # deprecated due to randomly switch between padding and non-padding
        )

        """Whether using padding for test"""
        self.is_padding_for_test: bool = False

        """ MultiMAE """
        self.decoder_depth: int = 4
        self.is_inference_with_no_depth: bool = False
        self.inputs = ["rgb", "depth"]
        self.outputs = ["semseg"]
        self.decoder_main_tasks: List[List[str]] = [["rgb"]]

        """Data Augmentation"""
        self.data_augmentation_version: int = 2

        self.ckpt_path: str = None
        self.description: str = ""  # Override this
        self.embed_dim: int = 6144

        self.lr: float
        self.end_lr: float = 1e-11
        self.lr_scale: int
        self.lr_power: float = 0.9

        self.save_checkpoints_after_each_n_epochs: int = 10

        self.weight_decay = 0.05
        self.num_workers = 2
        self.warm_up_epoch = 0

        self.betas: Tuple[float, float] = (0.9, 0.999)

        self.input_patch_size: int = 16
        self.output_patch_size: int = 16

        self.batch_size: int
        self.val_batch_size: int
        self.nepochs: int

        if run_type in [RUN_TYPE.COLAB, RUN_TYPE.KAGGLE]:
            self.em = ExperimentManager(
                self.experiment_name,
                self.pickle_dir_path,
                self.experiment_dir_path,
            )
            # self.em.clean()
            if self.em.latest_epoch is not None:
                self.ckpt_path: str = os.path.join(
                    self.experiment_dir_path,
                    self.experiment_name,
                    f"checkpoint_{self.em.latest_epoch}.pt",
                )

            self.device = get_device()
            self.cpu_device = torch.device("cpu")
        elif run_type == RUN_TYPE.HUGGINGFACE:
            # when using this in production, please specify "epoch"
            self.ckpt_path: str = get_production_ckpt_path(self.experiment_name, epoch)


class cfgv4_1_17(base_cfg):
    def __init__(self, epoch: int = None):
        super().__init__(
            epoch, experiment_name="exp_v4.0.19", datasets_set=1, run_type=run_type
        )

        self.description = "DAv2-Base"
        self.accum_iter = 2  # <---------------

        """Learning rate"""
        self.lr = 1e-5
        self.end_lr = 1e-11
        self.lr_scale = 100

        self.batch_size = 32  # <---------------
        self.val_batch_size = 300  # <---------------
        self.nepochs = 200

        self.decoder_depth = 4
        self.decoder_main_tasks = [["rgb", "depth"]]

        self.data_augmentation_version = 2
        self.save_checkpoints_after_each_n_epochs = 5


class cfg_test(base_cfg):
    def __init__(self, epoch: int = None):
        super().__init__(
            epoch, experiment_name="exp_v4.0.19", datasets_set=1, run_type=run_type
        )

        self.description = "DAv2-Base"
        self.accum_iter = 2  # <---------------

        self.image_size = 448
        self.test_image_size = 448
        self.embed_dim = 6144 * 4
        self.input_patch_size = 32
        self.output_patch_size: int = 64

        """Learning rate"""
        self.lr = 1e-5
        self.end_lr = 1e-11
        self.lr_scale = 100

        self.batch_size = 32  # <---------------
        self.val_batch_size = 300  # <---------------
        self.nepochs = 200

        self.decoder_depth = 4
        self.decoder_main_tasks = [["rgb", "depth"]]

        self.data_augmentation_version = 2
        self.save_checkpoints_after_each_n_epochs = 5
