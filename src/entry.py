# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from model import GraphFormer

from data import GraphDataModule, get_dataset
from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os

from pytorch_lightning.plugins import DDPPlugin

def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    #add weak_ensemble argument
    parser.add_argument('--weak_ensemble', type=int, default=0)
    parser = pl.Trainer.add_argparse_args(parser)
    

    parser = GraphFormer.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.plugins = DDPPlugin(find_unused_parameters=True)
    args.max_steps = args.tot_updates + 1
    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    dm = GraphDataModule.from_argparse_args(args)

    # ------------
    # model
    # ------------
    # pretrained_model = GraphFormer.load_from_checkpoint(
    #         "/da1/G2GT/src/lightning_logs/version_53/checkpoints/epoch=70-step=426440.ckpt",
    #         strict=False,
    #         n_layers=args.n_layers,
    #         head_size=24,
    #         hidden_dim=args.hidden_dim,
    #         attention_dropout_rate=args.attention_dropout_rate,
    #         dropout_rate=args.dropout_rate,
    #         intput_dropout_rate=args.intput_dropout_rate,
    #         weight_decay=args.weight_decay,
    #         ffn_dim=args.ffn_dim,
    #         dataset_name=dm.dataset_name,
    #         warmup_updates=args.warmup_updates,
    #         tot_updates=args.tot_updates,
    #         peak_lr=args.peak_lr,
    #         end_lr=args.peak_lr,
    #         edge_type=args.edge_type,
    #         multi_hop_max_dist=args.multi_hop_max_dist,
    #         flag=args.flag,
    #         flag_m=args.flag_m,
    #         flag_step_size=args.flag_step_size,
            
    #     )




    if args.checkpoint_path != '':
        
        model = GraphFormer.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            n_layers=args.n_layers,
            head_size=args.head_size,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.peak_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
            inference_path = args.inference_path,
            weak_ensemble = args.weak_ensemble,
            
        )


    else:
        model = GraphFormer(
            n_layers=args.n_layers,
            head_size=args.head_size,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
            inference_path = args.inference_path,
            weak_ensemble = args.weak_ensemble,
            
        )
    # pretrained_dict = pretrained_model.state_dict()
    # model_dict = model.state_dict()
    # key_list = ["atom_encoder.weight","edge_encoder.weight","rel_pos_encoder.weight","in_degree_encoder.weight","out_degree_encoder.weight"
    # ,"atom_edge_encoder.weight",'centrality_encoder.weight', 'lpe_linear.weight', 'lpe_linear.bias', 'lpe_linear3.weight', 'lpe_linear3.bias', 'position.pe']
    # for key in key_list:
    #     model_dict[key] = pretrained_dict[key]
    # model.load_state_dict(model_dict)




    
    if not args.test and not args.validate:
        print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # ------------
    # training
    # ------------
    metric = 'valid_' + get_dataset(dm.dataset_name)['metric']
    dirpath = args.default_root_dir + f'/lightning_logs/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        #filename=dm.dataset_name,
        save_last=True,
        every_n_train_steps=500,
        #every_n_epochs = 2,
    )
    
    
    if not args.test and not args.validate and os.path.exists(dirpath + '/last.ckpt'):
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('args.resume_from_checkpoint', args.resume_from_checkpoint)
        
        
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))


    if args.test:
        result = trainer.test(model, datamodule=dm)
        pprint(result)
    elif args.validate:
        result = trainer.validate(model, datamodule=dm)
        pprint(result)
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
