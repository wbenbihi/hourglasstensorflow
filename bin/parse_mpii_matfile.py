import os
import json
import scipy.io
import click

from utils.mpii_mat_handler import parse_act, parse_img_train, parse_single_person, parse_annolist, parse_video_list


@click.command()
@click.option('--filename', help='Path of .mat file')
@click.option('--output', default='', help='Output folder')
def main(filename, output):
    """

    """
    click.echo(f'Lauching {filename} parsing...')
    click.echo(f'Reading {filename}...')
    # Opening Data
    mat = scipy.io.loadmat(filename, struct_as_record=False)
    release_mat = mat['RELEASE'][0][0]
    
    # Parsing mat file
    click.echo(f'Parsing annolist 1/5...')
    annolist = parse_annolist(release_mat.__dict__.get('annolist')[0])
    click.echo(f'Parsing act 2/5...')
    act = parse_act(release_mat.__dict__.get('act'))
    click.echo(f'Parsing single_person 3/5...')
    single_person = parse_single_person(release_mat.__dict__.get('single_person'))
    click.echo(f'Parsing img_train 4/5...')
    img_train = parse_img_train(release_mat.__dict__.get('img_train'))
    click.echo(f'Parsing video_list 5/5...')
    video_list = parse_video_list(release_mat.__dict__.get('video_list')[0])

    # Saving to JSON
    list_to_save = [
        (annolist, 'annolist.json'),
        (act, 'act.json'),
        (single_person, 'single_person.json'),
        (img_train, 'img_train.json'),
        (video_list, 'video_list.json')
    ]
    for j, f in list_to_save:
        with open(os.path.join(output, f), 'w', encoding='utf8') as fp:
            click.echo(f'Saving {os.path.join(output, f)}...')
            json.dump(j, fp, ensure_ascii=False)





if __name__ == "__main__":
    main()