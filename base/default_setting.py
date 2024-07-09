# -*- coding:utf-8 -*-

def add_server_specific_setting(parser):
    device_dict = {
        1: server1_config,
    }
    temp_args, _ = parser.parse_known_args()
    server_id = temp_args.server_id
    dataset = temp_args.dataset
    return device_dict[server_id](parser, dataset)

def server1_config(parser, dataset):
    if dataset == 'crop_resize_400x64x400_labeled':
        parser.add_argument('--image_npy_root',
                            default='/home/Data/huangkun/Choroid_Diffusion_3D/selected_data_for_choroid_diffusion/labeled_data/crop_resize_400x64x400/OCTv2/cubes_npy')
        parser.add_argument('--label_npy_root',
                            default='/home/Data/huangkun/Choroid_Diffusion_3D/selected_data_for_choroid_diffusion/labeled_data/crop_resize_400x64x400/labels/cubes_npy')
        parser.add_argument('--train_data_list', default=['1', '2', '94', '95', '215'])
        parser.add_argument('--test_data_list', default=['3', '92', '93', '234', '235'])
        parser.add_argument('--result_root', type=str, default='/home/Data/huangkun/Choroid_Diffusion_3D/result_ptl/crop_resize_400x64x400')

    if dataset == 'crop_move_400x128x400_up5_labeled':
        parser.add_argument('--image_npy_root',
                            default='/home/Data/huangkun/Choroid_Diffusion_3D/selected_data_for_choroid_diffusion/labeled_data/crop_move_400x128x400_up5/OCTv2/cubes_npy')
        parser.add_argument('--label_npy_root',
                            default='/home/Data/huangkun/Choroid_Diffusion_3D/selected_data_for_choroid_diffusion/labeled_data/crop_move_400x128x400_up5/labels/cubes_npy')
        parser.add_argument('--train_data_list', default=['1', '2', '94', '95', '215'])
        parser.add_argument('--test_data_list', default=['3', '92', '93', '234', '235'])
        parser.add_argument('--result_root', type=str, default='/home/Data/huangkun/Choroid_Diffusion_3D/result_ptl/crop_move_400x128x400_up5')

    if dataset == '400x640x400':
        parser.add_argument('--image_npy_root',
                            default='/home/Data/huangkun/Choroid_Diffusion_3D/selected_data_for_choroid_diffusion/labeled_data/OCTv2/cubes_npy')
        parser.add_argument('--label_npy_root',
                            default='/home/Data/huangkun/Choroid_Diffusion_3D/selected_data_for_choroid_diffusion/labeled_data/labels/cubes_npy')
        parser.add_argument('--train_data_list', default=['1', '2', '94', '95', '215'])
        parser.add_argument('--test_data_list', default=['3', '92', '93', '234', '235'])
        parser.add_argument('--result_root', type=str, default='/home/Data/huangkun/Choroid_Diffusion_3D/result_ptl/400x640x400')

    if dataset == 'semi_2d':
        parser.add_argument('--image_npy_root',
                            default='/home/Data/huangkun/Choroid_Diffusion_3D/selected_data_for_choroid_diffusion/labeled_data/OCTv2/cubes_npy')
        parser.add_argument('--label_npy_root',
                            default='/home/Data/huangkun/Choroid_Diffusion_3D/selected_data_for_choroid_diffusion/labeled_data/labels/cubes_npy')
        parser.add_argument('--train_data_list', default=['1', '2', '94', '95', '215'])
        parser.add_argument('--test_data_list', default=['3', '92', '93', '234', '235'])
        parser.add_argument('--result_root', type=str, default='/home/Data/huangkun/Choroid_Diffusion_3D/result_ptl/400x640x400')