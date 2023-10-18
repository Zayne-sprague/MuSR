import sys
import time
from pathlib import Path
import subprocess
import argparse
import shutil

ROOT_FOLDER = Path(__file__).parent
DATA_FOLDER = ROOT_FOLDER / 'output_datasets'
TMP_FOLDER = DATA_FOLDER / 'tmp'

DATA_FOLDER.mkdir(exist_ok=True, parents=True)
TMP_FOLDER.mkdir(exist_ok=True, parents=True)

files = {
    'musr_murder_mystery': {'id': 'qfmhyuzyzayfr7vszfz6oue8smpi30np', 'name': 'musr_murder_mystery',
                            'local_filename': 'musr_murder_mystery'},
    'musr_team_allocation': {'id': 'caely78x5jskjy4wlln6vq51ebd9uw7g', 'name': 'musr_team_allocation',
                            'local_filename': 'musr_team_allocation'},
    'musr_object_placements': {'id': 'pd6uej3vji9geh992hjkgqo8hj6r4obo', 'name': 'musr_object_placements'},

}


def install_file(file_id, name, local_name: str = None):
    """
    :param file_id: The BOX file id for the current file being downloaded.
    :param name: The name of the file in the BOX folder (as well as the name of the folder to be used locally)
    :return: N/A
    """

    subprocess.run(['curl', '-L', f'https://utexas.box.com/shared/static/{file_id}.json', '--output',
                    str(DATA_FOLDER / f'{name}.json')])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ignore_list', '-i', choices=list(files.keys()), type=str, nargs='+',
        help='Any file you may not want to download and put into the output_datasets folder.'
    )
    parser.add_argument(
        '--allow_list', '-a', choices=list(files.keys()), type=str, nargs='+',
        help='Any of the files you want to download and put into the output_datasets folder.'
    )
    parser.add_argument('--show_files', '-s', action='store_true', help='List available files to download.')

    args = parser.parse_args()

    ignore_list = args.ignore_list
    allow_list = args.allow_list
    show_files = args.show_files

    if show_files:
        print("==== Files you can download are ===")
        for f in files.keys():
            print(f'\t{f}')
        sys.exit(0)

    files_to_download = list(files.keys())
    if ignore_list is not None:
        files_to_download = [x for x in files_to_download if x not in ignore_list]
    if allow_list is not None:
        files_to_download = [x for x in files_to_download if x in allow_list]

    assert len(files_to_download) > 0, \
        'No files were found using the ignore and allow list parameters!'

    print('=====   INFO   ======')
    print("These are big files and it may take awhile to download.")
    print('=====================')

    for file in files_to_download:
        install_file(files[file]['id'], files[file]['name'], files[file].get('local_filename', None))

    if TMP_FOLDER.exists():
        shutil.rmtree(str(TMP_FOLDER))