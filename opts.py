import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    #data input settings
    parser.add_argument('--input_json', type=str, default='data/coco.json',
                        help='path to josn file containing additional info and vocab')
    parser.add_argument('--input_h5', type=str, default='data/coco.h5')
    parser.add_argument('--start_from', type=str, default=None,
                        help="continue training from this path. Path must contain files saved by previous training process")

    #model params
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--att_feat_size',type=int, default=1024)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--use_selector', type=bool, default=True)

    # Optimization: General
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_per_img', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=0.1)

    # Optimization: for the language model
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1)
    parser.add_argument('--learning_rate_decay_every', type=int, default=3)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0)

    #Evaluation/Checkpoint
    parser.add_argument('--losses_log_every', type=int, default=25)
    parser.add_argument('--save_checkpoint_every', type=int, default=2500)
    parser.add_argument('--language_eval', type=int, default=0)
    parser.add_argument('--load_best_score', type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str, default='save')
    parser.add_argument('--val_images_use', type=int, default=3200)

    #mis
    parser.add_argument('--id', type=str, default="")
    parser.add_argument('--train_only', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=224)

    args = parser.parse_args()

    #check if args are valid


    return args



