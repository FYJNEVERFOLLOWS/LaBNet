import os
import datetime

def generate_list(input_dir, ouput_dir, data_split):
    date = datetime.datetime.now().strftime("%m%d")
    os.makedirs(ouput_dir, exist_ok=True)
    output_wav_lst = os.path.join(ouput_dir, '{}_{}.lst'.format(data_split, date))
    with open(output_wav_lst, 'w', encoding='utf-8') as output_f:
        save_num = 0
        if os.path.exists(input_dir):
            subfolder_list = os.listdir(input_dir)
            subfolder_list = [os.path.join(input_dir, subfolder_name) for subfolder_name in subfolder_list]
            for subfolder in subfolder_list:
                file_path_list = os.listdir(subfolder)
                for file_path in file_path_list:
                    output_f.write(subfolder+ os.sep+ file_path.strip()+'\n')
                    save_num += 1
                print('Finish save: {}'.format(save_num))


if __name__ == '__main__':
    # data_split = 'test-clean_wer'
    data_split = 'test-clean_0226'
    # data_split = 'dev-clean'
    # data_split = 'train-clean-100'
    # input_dir = f'/local02/fuyanjie/Libri-SIM/jsons/{data_split}'
    input_dir = f'/CDShare3/Libri-SIM/jsons_0129/{data_split}'
    ouput_dir = '/Work21/2021/fuyanjie/pycode/LaBNet/data/exp_list'
    generate_list(input_dir, ouput_dir, data_split)