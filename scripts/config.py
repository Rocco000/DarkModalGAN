import yaml
import json
from pathlib import Path

class Config:
    """
    This class handle the YAML file.
    """
    def __init__(self, yaml_path:Path):
        """
        :param yaml_path: a path to the YAML file.
        """
        if not yaml_path.exists() or not yaml_path.is_file():
            raise FileNotFoundError(f"{yaml_path} doesn't exist!")
        
        self.yaml_path = yaml_path
        self.config = None

    def load_yaml(self):
        """
        Read the YAML file.
        """
        with open(self.yaml_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_dataset_path(self) -> Path:
        dataset_path = Path(self.config["dataset_path"])

        if not dataset_path.exists() or not dataset_path.is_file():
            raise FileNotFoundError(f"{dataset_path} doesn't exist")
        
        return dataset_path

    def get_batch_size(self) -> int:
        return self.config["batch_size"]
    
    def get_num_epoch(self) -> int:
        return self.config["num_epoch"]
    
    def get_lr(self):
        return float(self.config["lr"])
    
    def get_lambda(self):
        return self.config["lambda"]
    
    def get_z_dim(self) -> int:
        return self.config["z_dim"]
    
    def get_critic_iteration(self) -> int:
        return self.config["critic_iteration"]
    
    def get_img_size(self) -> int:
        return self.config["img_size"]
    
    def get_feature_map(self) -> int:
        return self.config["feature_map"]
    
    def get_embedding_size(self) -> int:
        return self.config["embedding_size"]
    
    def get_n_encoder_block(self) -> int:
        return self.config["n_encoder_block"]
    
    def get_n_decoder_block(self) -> int:
        return self.config["n_decoder_block"]
    
    def get_n_head(self) -> int:
        return self.config["n_head"]
    
    def get_dropout(self):
        return self.config["dropout"]
    
    def get_intermediate_dim(self) -> int:
        return self.config["intermediate_dim"]
    
    def get_label_embeddig_size(self) -> int:
        return self.config["label_embedding_size"]
    
    def get_origin_mapping(self) -> dict:
        """
        Read the one-hot encoding mapping for origin feature and flip the mapping.
        
        :return: the flipped one-hot encoding mapping.
        """
        origin_mapping_path =  Path(self.config["origin_mapping"])

        if not origin_mapping_path.exists():
            raise FileNotFoundError(f"{origin_mapping_path} doesn't exist")

        origin_mapping = None
        with open(origin_mapping_path, "r") as file:
            origin_mapping = json.load(file)

        reverse_origin_mapping = {tuple(v): k for k, v in origin_mapping.items()}

        return reverse_origin_mapping
    
    def get_destination_mapping(self) -> dict:
        """
        Read the one-hot encoding mapping for destination feature and flip the mapping.
        
        :return: the flipped one-hot encoding mapping.
        """
        destination_mapping_path =  Path(self.config["destination_mapping"])

        if not destination_mapping_path.exists():
            raise FileNotFoundError(f"{destination_mapping_path} doesn't exist")
        
        destination_mapping = None
        with open(destination_mapping_path, "r") as file:
            destination_mapping = json.load(file)

        reverse_destination_mapping = {tuple(v): k for k, v in destination_mapping.items()}

        return reverse_destination_mapping
    
    def get_micro_category_mapping(self) -> dict:
        """
        Read the one-hot encoding mapping for micro-category feature and flip the mapping.
        
        :return: the flipped one-hot encoding mapping.
        """
        micro_category_mapping_path =  Path(self.config["micro_category_mapping"])

        if not micro_category_mapping_path.exists():
            raise FileNotFoundError(f"{micro_category_mapping_path} doesn't exist")
        
        micro_category_mapping = None
        with open(micro_category_mapping_path, "r") as file:
            micro_category_mapping = json.load(file)

        reverse_micro_category_mapping = {tuple(v): k for k, v in micro_category_mapping.items()}

        return reverse_micro_category_mapping

    def get_checkpoint_path(self) -> Path:
        """
        Returns the file path to the checkpoint file.

        :return: Path object representing the checkpoint file path.
        """
        return Path(self.config["checkpoint_path"])

    def get_gen_weights(self, epoch:int) -> Path:
        """
        Returns the file path to the generator weigths for given epoch.

        :param epoch: the epoch number at which the generator weights are saved.
        :return: Path object representing the generator weights file path.
        :raises FileNotFoundError: If the generator weights file does not exist for the given epoch.
        """
        gen_weights_path = Path(self.config["gen_path"]).joinpath(f"generator_weigths_{epoch}.pth")
        if not gen_weights_path.exists():
            raise FileNotFoundError(f"{gen_weights_path} doesn't exist")
        return gen_weights_path

    def get_critic_weights(self, epoch:int) -> Path:
        """
        Returns the file path to the critic weigths for given epoch.

        :param epoch: the epoch number at which the critic weights are saved.
        :return: Path object representing the critic weights file path.
        :raises FileNotFoundError: If the critic weights file does not exist for the given epoch.
        """
        critic_weights_path = Path(self.config["critic_path"]).joinpath(f"critic_weigths_{epoch}.pth")
        if not critic_weights_path.exists():
            raise FileNotFoundError(f"{critic_weights_path} doesn't exist")
        return critic_weights_path

    def get_gen_opt_state(self, epoch:int) -> Path:
        """
        Returns the file path to the generator optimizer state for given epoch.

        :param epoch: the epoch number at which the generator optimizer state is saved.
        :return: Path object representing the generator optimizer state file path.
        :raises FileNotFoundError: If the generator optimizer state file does not exist for the given epoch.
        """
        gen_opt_path = Path(self.config["gen_opt_path"]).joinpath(f"gen_optimizer_state_{epoch}.pth")
        if not gen_opt_path.exists():
            raise FileNotFoundError(f"{gen_opt_path} doesn't exist")
        return gen_opt_path

    def get_critic_opt_state(self, epoch:int) -> Path:
        """
        Returns the file path to the critic optimizer state for given epoch.

        :param epoch: the epoch number at which the critic optimizer state is saved.
        :return: Path object representing the critic optimizer state file path.
        :raises FileNotFoundError: If the critic optimizer state file does not exist for the given epoch.
        """
        critic_opt_path = Path(self.config["critic_opt_path"]).joinpath(f"critic_optimizer_state_{epoch}.pth")
        if not critic_opt_path.exists():
            raise FileNotFoundError(f"{critic_opt_path} doesn't exist")
        return critic_opt_path
    
    def get_writer_path_real(self) -> Path:
        writer_real = Path(self.config["writer_real"])

        if not writer_real.exists():
            writer_real.mkdir(parents=True)
            print(f"Created {writer_real}")

        return writer_real
    
    def get_writer_path_fake(self) -> Path:
        writer_fake = Path(self.config["writer_fake"])

        if not writer_fake.exists():
            writer_fake.mkdir(parents=True)
            print(f"Created {writer_fake}")

        return writer_fake
    
    def get_user_name(self) -> str:
        return self.config["user_name"]
