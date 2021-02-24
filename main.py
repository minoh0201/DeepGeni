import os
import argparse
import data_loader
import config
import wGAN_augmentor

from experimentor import Experimentor

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lsocv", help="Leave one study out cross validation", action='store_true')
    parser.add_argument("--cv", help="cross validation", action='store_true')
    parser.add_argument("--expname", help="Experiment name to create new directory under the results", type=str, default="baseline")

    args = parser.parse_args()
    print(args)
    
    # Load data
    X, y, ids, features = data_loader.load_OTUprofile_and_labels()
    Xs, ys, studies = data_loader.split_studies(X=X, y=y, ids=ids)

    # Baseline without feature selection
    #exp = Experimentor(Xs=Xs, ys=ys, studies=studies, name='baseline', feature_selection=False)
    #exp.classify()

    # With feature selection
    # num_max_features = 32
    # args.expname = args.expname + '_' + str(num_max_features)
    # exp = Experimentor(Xs=Xs, ys=ys, studies=studies, name=args.expname, feature_selection=True, num_max_features=num_max_features)
    # exp.classify()

    # Simple AE
    # dims=[128, 64]
    # args.expname = args.expname + '_' + '_'.join([str(x) for x in dims])
 
    # exp = Experimentor(Xs=Xs, ys=ys, studies=studies, name=args.expname, feature_selection=True, num_max_features=256)
    # exp.ae(dims=dims, patience=30)
    # exp.classify()

    # Aug and Eval
    # exp = Experimentor(Xs=Xs, ys=ys, studies=studies, name=args.expname, feature_selection=True, num_max_features=256)
    # aug_rates = [0.5, 1, 2, 4, 8, 16]

    # # Set num clusters and gans
    # exp.viz_wss() 
    # exp.num_clusters = [5, 5, 5, 5]
    # exp.num_GANs = [5, 7, 7, 9]

    # # call augmentor  
    # wGAN_augmentor.deepbiogen(exp = exp, 
    #                             aug_rates=aug_rates, 
    #                             num_epochs=6000, 
    #                             batch_size=128, 
    #                             sample_interval=2000,
    #                             save_all_data=False)
    
    # exp.classify_with_DBG()

    if args.expname == "FS-DBG-AE":
        # Aug and AE
        exp.viz_wss()
        exp.num_clusters = [5, 5, 5, 5]
        exp.num_GANs = [5, 7, 7, 9]

        # call augmentor  
        wGAN_augmentor.deepbiogen(exp = exp, 
                                    aug_rates=aug_rates, 
                                    num_epochs=6000, 
                                    batch_size=128, 
                                    sample_interval=2000,
                                    save_all_data=False)

        exp.ae(augmented_training=True)
        exp.classify()
        exp.classify_with_DBG()

    if args.expname == "FS-AE-DBG":
        # AE and Aug
        dims=[128, 64]    
        exp = Experimentor(Xs=Xs, ys=ys, studies=studies, name=args.expname, feature_selection=True, num_max_features=256)
        exp.ae(dims=dims, patience=30)
        
        exp.viz_wss()
        exp.num_clusters = [6, 4, 5, 7]
        exp.num_GANs = [2, 7, 5, 5]
        # call augmentor  
        wGAN_augmentor.deepbiogen(exp = exp, 
                                    aug_rates=aug_rates, 
                                    num_epochs=6000, 
                                    batch_size=128, 
                                    sample_interval=2000,
                                    save_all_data=False)
    
        exp.classify_with_DBG()
    
    print('The End')
