# Helper script to extract speaker embeddings (x-vectors)
import torchaudio
import glob
import pickle
import click

@click.command()
@click.argument('folder', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=False))
def xvector_extract(folder: str, output: click.Path):
    xvector_files = glob.glob(f'{folder}/**/*.scp', recursive=True)

    print(f'Found {len(xvector_files)} scp files. Loading them ...')
    d = {}

    for fname in xvector_files:
        try:
            d1 = {u: d for u,d in torchaudio.kaldi_io.read_vec_flt_scp(fname)}
            print(f'Reading {len(d1)} xvectors from {fname} ...')
            
            d.update(d1)
        except Exception as e:
            print(f'Reading {fname} failed!\n{e}')

    print(f'Finished reading {len(d)} xvectors!')

    foutname = f'{output}.pkl'
    print(f'Writing aggregated xvector file to {foutname} ...')
    with open(foutname, mode='wb') as fout:
        pickle.dump(d, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Done!')

if __name__ == "__main__":
    xvector_extract()
