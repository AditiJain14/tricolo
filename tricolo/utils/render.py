"""Python wrapper for rendering voxels using stk.
Reference: https://github.com/ruanyyyyyyy/text2shape/blob/master/lib/render.py
"""

import os
import subprocess
import argparse
import glob
import numpy as np

import tricolo.utils.nrrd_rw as nrrd_rw
from tricolo.utils.data_io import get_voxel_file


def render_model_id(dataset, model_ids, out_dir, check=True):
    """Render models based on their model IDs.
    Args:
        model_ids: List of model ID strings.
        nrrd_dir: Directory to write the NRRD files to.
        out_dir: Directory to write the PNG files to.
        check: Check if the output directory already exists and provide a warning if so.
    """
    if dataset == 'primitives':
        categories = [model_id.split('_')[0] for model_id in model_ids]
    elif dataset == 'shapenet':
        categories = [None] * len(model_ids)
    else:
        raise ValueError('Please use a valid dataset.')

    if check is True:
        if os.path.isdir(out_dir):
            print('Output directory:', out_dir)
            input('Output render directory exists! Continue?')
        else:
            os.makedirs(out_dir)
    else:
        os.makedirs(out_dir, exist_ok=True)

    nrrd_files = []
    for category, model_id in zip(categories, model_ids):
        nrrd_files.append(get_voxel_file(dataset, category, model_id))

    txt_filepath = os.path.join(out_dir, 'nrrd_filenames.txt')
    with open(txt_filepath, 'w') as f:
        for outfile in nrrd_files:
            f.write(outfile + '\n')
    # print('Filenames written to {}'.format(txt_filepath))

    # render_nrrd(txt_filepath, out_dir, check=False)
    # else:
    #     voxel_tensors = []
    #     for model_id in model_ids:
    #         voxel_tensors.append(load_voxel(None, model_id))
    #     render_voxels(voxel_tensors, nrrd_dir, out_dir)



def render_nrrd(nrrd,
                out_dir,
                turntable=False,
                turntable_step=10,
                compress_png=False,
                check=True):
    """Render NRRD files.
    Args:
        nrrd: An NRRD filename or txt file containing the NRRD filenames.
        out_dir: Output directory for the NRRD files.
        turntable: Whether or not to render a turntable.
        turntable_step: Number of degrees between each turntable step.
        compress_png: Whether to compress the png.
        check: Check if the output directory already exists and provide a warning if so.
    """
    if (check is True) and os.path.isdir(out_dir):
        input('Output render directory exists! Continue?')
    else:
        os.makedirs(out_dir, exist_ok=True)

    if turntable is True:
        turntable_str = '--render_turntable --turntable_step {}'.format(turntable_step)
    else:
        turntable_str = ''

    if compress_png is True:
        compress_png_str = '--compress_png'
    else:
        compress_png_str = ''

    render_command = [
        'node',
        '--max-old-space-size=24000',
        '{}/ssc/render-voxels.js'.format('path/to/your/scene-toolkit/'),
        '--input',
        nrrd,
        '--output_dir',
        out_dir,
        # turntable_str,
        # compress_png_str,
    ]

    # subprocess.run is only supported on Python 3.5+
    # Otherwise, using subprocess.call
    subprocess.run(render_command, stdout=subprocess.PIPE)



def batch_render_voxels(voxel_filenames, dest_dir, write_txt=False, voxel_processor=None):
    """Render a list of voxel tensors from npy files.
    Example: voxel_filenames = ['/tmp/voxel.npy']
    Args:
        model_list: List of model filenames (such as in .npy format).
        write_txt: Boolean for whether to write the generated NRRD filenames to a txt file. The file
                is written to the directory containing the FIRST file in voxel_filenames.
        voxel_processor: Function that postprocesses the loaded voxels
    Returns:
        outfiles: List of written files.
    """
    if len(voxel_filenames) > 9999:
        raise NotImplementedError('Cannot render %d images' % len(voxel_filenames))

    filename_ext = '.nrrd'
    voxels = voxel_filenames
    num_renderings = len(voxels)
    outfiles = []
    for voxel_idx, voxel_f in enumerate(voxels):
        if (voxel_idx + 1) % 10 == 0:
            print('Rendering %d/%d.' % (voxel_idx + 1, num_renderings))

        outfile = os.path.join(dest_dir, os.path.splitext(voxel_f)[0].split('/')[-1] + filename_ext)
        voxel_tensor = np.load(voxel_f)
        if voxel_processor is not None:
            voxel_tensor = voxel_processor(voxel_tensor)

        nrrd_rw.write_nrrd(voxel_tensor, outfile)
        outfiles.append(outfile)
        if voxel_idx == 15:
            break

    if write_txt is True:
        txt_filename = 'nrrd_filenames.txt'
        txt_filepath = os.path.join(dest_dir, txt_filename)
        with open(txt_filepath, 'w') as f:
            for outfile in outfiles:
                f.write(outfile + '\n')
        print('Filenames written to {}'.format(txt_filepath))
    return outfiles

def batch_render(voxel_dir, dest_dir, voxel_processor=None):
    """Render a bunch of voxel tensors stored as npy files in the voxel_dir.
    Args:
        voxel_dir: A directory containing voxel tensors stored in npy files.
        voxel_processor: Function that processes the voxels before rendering
    """
    npy_files = glob.glob(os.path.join(voxel_dir, '*.npy'))
    batch_render_voxels(npy_files, dest_dir, write_txt=True, voxel_processor=voxel_processor)
    render_nrrd('logs/generation/results/nrrd_output/nrrd_filenames.txt', dest_dir, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('voxel_dir', help='Directory of saved npy files')
    parser.add_argument('dest_dir', help='Directory to save to)')

    args = parser.parse_args()
    postprocessor = None
    batch_render(args.voxel_dir, args.dest_dir, postprocessor)


if __name__ == '__main__':
    main()
