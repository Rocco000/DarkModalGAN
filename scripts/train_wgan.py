from pathlib import Path
import pandas as pd
import csv
from tqdm import tqdm
from transformers import BertTokenizer # For tokenization
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import traceback

from collections import defaultdict

# For logging
from torch.utils.tensorboard import SummaryWriter

import dagshub
import mlflow
import mlflow.pytorch

# Local import
from models import Generator, Critic, initialize_weights, gradient_penalty
from dataset_class import MultimodalDataset
from config import Config
from log_functions import log_categorical, log_scalar_metrics, log_img_metrics, log_txt_metrics

drug_terms = [
    "cannabidiol",
    "cannabidiolic acid",
    "cannabidivarin",
    "cannabigerol",
    "cannabinol",
    "concentrate",
    "ak47",
    "shake",
    "tetrahydrocannabinol",
    "tetrahydrocannabinolic acid",
    "rick simpson oil",
    "nandrolone phenylpropionate",
    "trenbolone",
    "boldenone",
    "turinabol",
    "dihydrotestosterone",
    "ligandrol",
    "nutrobal",
    "ostarine",
    "human chorionic gonadotropin",
    "human growth hormone",
    "clostebol",
    "nandrolone",
    "androstenedione",
    "dimethyltryptamine",
    "lysergic acid diethylamide",
    "isolysergic acid diethylamide",
    "metaphedrone",
    "mephedrone",
    "nexus",
    "psilacetin",
    "mebufotenin",
    "psilocin",
    "methylenedioxymethamphetamine",
    "amphetamine",
    "methamphetamine",
    "oxycontin",
    "oxycodone",
    "acetylcysteine",
    # Drug name
    "k8",
    "rp15",
    "tramadol",
    "roxycodone",
    "nervigesic",
    "pregabalin",
    "carisoprodol",
    "alprazolam",
    "xanax",
    "anavar",
    "benzodiazepine",
    "cocaine",
    "clenbuterol",
    "benzocaine",
    "clomiphene",
    "crack",
    "marijuana",
    "hashish",
    "nbome",
    "hydroxycontin chloride",
    "ketamine",
    "heroin",
    "adderall",
    "sativa",
    "indica",
    "cookie",
    "mushroom",
    "dihydrocodeine",
    "psilocybin"
]

def initialize_bert_tokenizer() -> BertTokenizer:
    """
    Initialize the BertTokenizer for tokenization and add the domain-specific terms to the dictionary.
    :return: a BertTokenizer instance.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    new_tokens = [word for word in drug_terms if word not in tokenizer.vocab]

    tokenizer.add_tokens(new_tokens)

    return tokenizer

def word_piece_pruning(ds:pd.DataFrame, seq_len:int, tokenizer:BertTokenizer, domain_terms:list, threshold:int, save_path:Path) -> BertTokenizer:
    """
    Apply the WordPiece Pruning on BertTokenizer vocabulary and store a new vocabulary. It removes tokens with low frequency in the dataset and store the left tokens.

    :param ds: a DataFrame instance.
    :param seq_len: the maximum sequence length.
    :param tokenizer: a BertTokenizer instance.
    :param domain_terms: a list of domain terms to keep in the new vocabulary.
    :param threshold: a threshold to filter tokens.
    :param save_path: a Path to store the new vocabulary.
    :return: a BertTokenizer with a pruned vocabulary.
    """
    dataset_text = list(ds["full_text"])
    token_counts = defaultdict(int)

    # 1) Compute token frequency
    for sentece in dataset_text:
        token_ids = tokenizer.encode(sentece, add_special_tokens=False, max_length=seq_len, truncation=True)
        for tid in token_ids:
            token_counts[tid] += 1

    # 2) Filter tokens that appear less than threshold times
    keep_ids = []
    for tid, freq in token_counts.items():
        if freq >= threshold:
            keep_ids.append(tid)

    original_vocab = tokenizer.get_vocab() # {token:id}
    index2token = {v: k for k, v in original_vocab.items()} # {id:token}

    # 3) Keep special tokens forcibly: [CLS], [SEP], [UNK], [PAD], [MASK], etc.
    special_ids = [
        original_vocab.get("[CLS]", None),
        original_vocab.get("[SEP]", None),
        original_vocab.get("[UNK]", None),
        original_vocab.get("[PAD]", None),
        original_vocab.get("[MASK]", None)
    ]

    # 4) Keep domain terms
    for term in domain_terms:
        special_ids.append(original_vocab.get(term, None))
    
    for sid in special_ids:
        if sid is not None:
            keep_ids.append(sid)

    keep_ids = list(set(keep_ids))  # remove duplicates

    # 5) Define a new vocabulary
    new_token2id = {}
    new_id2token = []
    idx = 0

    # Add special tokens
    for sid in special_ids:
        if sid is not None and sid not in new_token2id:
            token_str = index2token[sid]
            new_token2id[token_str] = idx
            new_id2token.append(token_str)
            idx += 1

    # then keep the tokens that remain
    for tid in keep_ids:
        
        if tid in special_ids:
            continue  # already added

        token_str = index2token[tid]
        new_token2id[token_str] = idx
        new_id2token.append(token_str)
        idx += 1

    with open(save_path, "w", encoding="utf-8") as f:
        for token in new_id2token:
            f.write(token + "\n")

    return BertTokenizer(save_path, do_lower_case=True)

def load_checkpoint(config:Config):
  """
  Returns the checkpoint information.

  :param config: a Config instance.
  :return: the mlflow experiment id, epoch number, tensorboard step, generator weights path, critic weights path, generator optimizer state path, critic optimizer state path.
  """
  checkpoint_df = pd.read_csv(config.get_checkpoint_path(), dtype={"experiment_id":str, "epoch":int, "tensorboard_step":int})
  exp_id = checkpoint_df["experiment_id"][0]
  epoch = checkpoint_df["epoch"][0]
  tensorboard_step = checkpoint_df["tensorboard_step"][0]

  gen_weights = config.get_gen_weights(epoch)
  critic_weights = config.get_critic_weights(epoch)

  gen_opt_state = config.get_gen_opt_state(epoch)
  critic_opt_state = config.get_critic_opt_state(epoch)

  return exp_id, epoch, tensorboard_step, gen_weights, critic_weights, gen_opt_state, critic_opt_state

def log_metrics(
  config:Config,
  real_data:torch.Tensor,
  fake_img:torch.Tensor,
  fake_origin:torch.Tensor,
  fake_destination:torch.Tensor,
  fake_micro_category:torch.Tensor,
  fake_price:torch.Tensor,
  fake_crypto_price:torch.Tensor,
  step:int,
  writer_real:SummaryWriter,
  writer_fake:SummaryWriter
) -> None:
    """
    Log metrics on tensorboard

    :param config: a Config instance.
    :param real_data: a batch of real data.
    :param fake_img: a batch of generated images.
    :param fake_origin: a batch of generated origin.
    :param fake_destination: a batch of generated destination.
    :param fake_micro_category: a batch of generated micro-category.
    :param fake_price: a batch of generated price.
    :param fake_crypto_price: a batch of generated crypto price.
    :param step: the step value.
    :param writer_real: a SummaryWriter to log metrics for real data.
    :param writer_fake: a SummaryWriter to log metrics for generated data.
    """
    with torch.no_grad():
        fake_img = fake_img.cpu() # (N, 3, 224, 224)
        fake_origin = fake_origin.cpu() # (N, 31)
        fake_destination = fake_destination.cpu() # (N, 18)
        fake_micro_category = fake_micro_category.cpu() #(N, 19)
        fake_price = fake_price.cpu() # (N,)
        fake_crypto_price = fake_crypto_price.cpu() # (N,)

        # Log label
        writer_real.add_text("Labels", str(real_data["label"].tolist()), global_step=step)
        writer_fake.add_text("Labels", str(real_data["label"].tolist()), global_step=step)

        # Log TABULAR DATA
        log_categorical(config, real_data["tabular"], fake_origin, fake_destination, fake_micro_category, writer_real, writer_fake, step)
        log_scalar_metrics(real_data["tabular"], fake_price, fake_crypto_price, writer_real, writer_fake, step)

        # Log IMG metrics
        log_img_metrics(real_data["image"], fake_img, writer_real, writer_fake, step)

def run_train(
  config:Config,
  train_data:DataLoader,
  num_epoch:int,
  start_epoch:int,
  start_step:int,
  z_dim:int,
  lambda_value:int,
  generator:Generator,
  critic:Critic,
  critic_iteration:int,
  gen_opt:optim.Adam,
  critic_opt:optim.Adam,
  writer_real:SummaryWriter,
  writer_fake:SummaryWriter,
  save_path:Path,
  device
) -> None:
    """
    Train the conditional WGAN.

    :param config: a Config instance.
    :param train_data: a DataLoader instance to iterate over dataset batches.
    :param num_epoch: the number of epoch.
    :param start_epoch: the epoch check-point.
    :param start_step: the step check-point.
    :param z_dim: the dimension of noisy z vector.
    :param lambda_value: lambda value for gradient penalty.
    :param generator: a Generator instance.
    :param critic: a Critic instance.
    :param critic_iteration: the number of iteration to train the discriminator.
    :param gen_opt: the Adam optimizer for the generator.
    :parma critic_opt: the Adam optimizer for the critic.
    :param writer_real: a SummaryWriter to track model performance on real samples.
    :param writer_fake: a SummaryWriter to track model performance on fake samples.
    :param save_path: a path to log the experiment.
    :param device: to move the models on input device (GPU or CPU).
    """
    tensorboard_step = start_step

    # Set models in train mode
    generator.train()
    critic.train()

    log_points = [10, 20, 30]

    mlflow.set_experiment("Training_conditionalWGAN_V2")

    with mlflow.start_run() as run:
        for epoch in range(start_epoch, num_epoch):
            batch_idx = 0
            tau = max(0.1, 1.0 * (0.99 ** epoch)) # Decay the tau value

            for batch in tqdm(train_data, desc=f"Processing Epoch {epoch:02d}"):
                batch_on_device = {key: value.to(device) for key, value in batch.items()}
                cur_batch_size = batch_on_device["image"].shape[0]
                gp = None

                # TRAIN CRITIC --> max E[D(x)] - E[D(G(x))]
                for _ in range(critic_iteration):
                    # Generate noisy vector Z : (N, z_dim)
                    noise = torch.randn(cur_batch_size, z_dim).to(device)

                    # GENERATE FAKE DATA
                    fake_img, fake_origin, fake_destination, fake_micro_category, fake_price, fake_crypto_price = generator(noise, batch_on_device["label"], tau)

                    # 1) Concatenate tabular data
                    fake_tabular = torch.cat([fake_origin, fake_destination, fake_micro_category, fake_price, fake_crypto_price], dim=1) # (N, tabular_dim)
                    del fake_origin, fake_destination, fake_micro_category, fake_price, fake_crypto_price

                    critic_real = critic(batch_on_device["image"], batch_on_device["tabular"], batch_on_device["label"])
                    critic_fake = critic(fake_img, fake_tabular, batch_on_device["label"])

                    # Calculate the gradient penalty
                    gp = gradient_penalty(critic, batch_on_device, fake_img, fake_tabular, batch_on_device["label"], device)

                    # Since the critic tries to maximize but the optimization algorithm is design to minimize, apply - at the beginning of the formula
                    critic_loss = (-(torch.mean(critic_real) - torch.mean(critic_fake))) + lambda_value * gp

                    critic.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                # TRAIN GENERATOR --> max E[D(G(x))] --> min - E[D(G(x))]
                # Re-run the generator
                noise = torch.randn(cur_batch_size, z_dim).to(device) # (N, z_dim)
                fake_img, fake_origin, fake_destination, fake_micro_category, fake_price, fake_crypto_price = generator(noise, batch_on_device["label"], tau)

                fake_tabular = torch.cat([fake_origin, fake_destination, fake_micro_category, fake_price, fake_crypto_price], dim=1) # (N, tabular_dim)

                if batch_idx not in log_points:
                    del fake_origin, fake_destination, fake_micro_category, fake_price, fake_crypto_price

                output = critic(fake_img, fake_tabular, batch_on_device["label"])

                gen_loss = -torch.mean(output)
                generator.zero_grad()
                gen_loss.backward()
                gen_opt.step()

                # VALIDATION
                if batch_idx in log_points:
                    writer_fake.add_scalar("Critic/Gradient_Penalty", gp.item(), global_step=tensorboard_step)
                    log_metrics(config, batch, fake_img, fake_origin, fake_destination, fake_micro_category, fake_price, fake_crypto_price, tensorboard_step, writer_real, writer_fake)

                    tensorboard_step += 1

                batch_idx += 1

            # Log loss after each epoch
            mlflow.log_metric("CRITIC Loss", critic_loss.item(), step=epoch)
            mlflow.log_metric("GENERATOR Loss", gen_loss.item(), step=epoch)
            mlflow.log_metric("Tau", tau, step=epoch)
            writer_fake.add_scalar("Generator/Loss", gen_loss.item(), global_step=epoch)
            writer_fake.add_scalar("Generator/Tau", tau, global_step=epoch)
            writer_real.add_scalar("Critic/Loss", critic_loss.item(), global_step=epoch)

            # Store model configuration
            mlflow.pytorch.log_state_dict(generator.state_dict(), artifact_path=f"models/generator/generator_weigths_{epoch}")
            mlflow.pytorch.log_state_dict(critic.state_dict(), artifact_path=f"models/critic/critic_weigths_{epoch}")
            if epoch % 50 == 0 or epoch == (num_epoch-1):
                torch.save(generator.state_dict(), config.get_gen_weights_folder().joinpath(f"generator_weigths_{epoch}.pth"))
                torch.save(critic.state_dict(), config.get_critic_weights_folder().joinpath(f"critic_weigths_{epoch}.pth"))

            # Store the optimizer
            mlflow.pytorch.log_state_dict(gen_opt.state_dict(), artifact_path=f"optimizers/gen_opt/gen_optimizer_state_{epoch}")
            mlflow.pytorch.log_state_dict(critic_opt.state_dict(), artifact_path=f"optimizers/critic_opt/critic_optimizer_state_{epoch}")
            if epoch % 50 == 0 or epoch == (num_epoch-1):
                torch.save(gen_opt.state_dict(), config.get_gen_opt_folder().joinpath(f"gen_optimizer_state_{epoch}.pth"))
                torch.save(critic_opt.state_dict(), config.get_critic_opt_folder().joinpath(f"critic_optimizer_state_{epoch}.pth"))

            with open(save_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                writer.writerow(["experiment_id", "epoch", "tensorboard_step"])
                writer.writerow([run.info.experiment_id, epoch, tensorboard_step])

if __name__ == "__main__":
    # Load configuration
    yaml_path = Path(input("Provide the YAML file path:\n"))
    config = Config(yaml_path)
    config.load_yaml()

    img_path = Path(input("Provide the image path:\n"))

    # Initialize dagshub
    dagshub.init(repo_owner=config.get_user_name(), repo_name="DarkModalGAN", mlflow=True)

    # Read dataset
    dataset_path = config.get_dataset_path()
    my_json_dataset = pd.read_json(dataset_path, orient="records", lines=True)

    # Hyperparameters
    BATCH_SIZE = config.get_batch_size()
    NUM_EPOCH = config.get_num_epoch()
    LEARNING_RATE = config.get_lr()
    LAMBDA = config.get_lambda()
    Z_DIM = config.get_z_dim()
    CRITIC_ITERATIONS = config.get_critic_iteration()
    LABEL_EMBEDDING_DIM = config.get_label_embeddig_size()
    print(f"CONFIGURATION: Batch = {BATCH_SIZE}; Epoch = {NUM_EPOCH}; Lambda= {LAMBDA}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # For tensorboard plotting
    writer_real = SummaryWriter(config.get_writer_path_real())
    writer_fake = SummaryWriter(config.get_writer_path_fake())

    # Create dataset class and define a DataLoader
    my_dataset = MultimodalDataset(ds=my_json_dataset, img_folder=img_path)
    train_dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create GENERATOR and CRITIC
    generator = Generator(device, num_classes=7, z_dim=Z_DIM, img_channels=3, img_size=224, feature_map=8, tabular_dim=[31, 18, 19], label_embedding_size=LABEL_EMBEDDING_DIM).to(device)

    # seq_len = 512
    critic = Critic(num_classes=7, tabular_dim=70, in_channels=3, img_size=224, feature_map=8, label_embedding_size=LABEL_EMBEDDING_DIM).to(device)

    # Define the optimizers
    gen_opt = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9)) # beta_1 = 0, beta_2 = 0.9 as in paper Improved Training of Wasserstein GANs
    critic_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9)) # beta_1 = 0, beta_2 = 0.9 as in paper Improved Training of Wasserstein GANs

    # Load checkpoint (optionally)
    checkpoint_flag = bool(int(input("Do you have a checkpoint for training? (1 = yes, 0 = no)\n")))
    exp_id = None
    epoch = 0
    tensorboard_step = 0

    if checkpoint_flag:
        exp_id, epoch, tensorboard_step, gen_weights, critic_weights, gen_opt_state, critic_opt_state = load_checkpoint(config)

        epoch += 1

        generator.load_state_dict(torch.load(gen_weights))
        critic.load_state_dict(torch.load(critic_weights))

        gen_opt.load_state_dict(torch.load(gen_opt_state))
        critic_opt.load_state_dict(torch.load(critic_opt_state))
    else:
        initialize_weights(generator)
        initialize_weights(critic)


    try:
        run_train(config, train_dataloader, NUM_EPOCH, epoch, tensorboard_step, Z_DIM, LAMBDA, generator, critic, CRITIC_ITERATIONS, gen_opt, critic_opt, writer_real, writer_fake, config.get_checkpoint_path(), device)
    except Exception as e:
        print(f"An error occurred:\n{e}")
        traceback.print_exc()
    finally:
        writer_fake.close()
        writer_real.close()
        print("Close writers")