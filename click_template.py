import click
from pathlib import Path
import logging
import sys

__version__ = 0.0
log = None

def getLogger(module_name, filename, stdout=None):
    format = '%(asctime)s [%(name)s:%(levelname)s] %(message)s'
    logging.basicConfig(filename=filename,
                        level=logging.DEBUG,
                        filemode='a',
                        format=format,
                        datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(module_name)

    # Add handler for stdout
    if stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(stdout)
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

@click.group()
@click.version_option(__version__)
@click.option('-v', '--verbose', count=True, show_default=True)
@click.option('-l', '--logfile', default=f'log.log', show_default=True)
@click.pass_context
def templateXXX(ctx, verbose, logfile):
    global log
    ctx.obj['workingdir'] = Path('.').absolute()
    loglevel = logging.WARNING
    if verbose > 1:
        loglevel = logging.INFO
    if verbose >= 2:
        loglevel = logging.DEBUG
    log = getLogger(__name__, ctx.obj['workingdir'].joinpath(logfile), stdout=loglevel)


def _subCMD(ctx, arg):
    print(f'Hello from subcommand {arg}')
    log.info(f"Running in cwd {ctx.obj['workingdir']}")


@templateXXX.command()
@click.argument('arg')
@click.pass_context
def subCMD(ctx, arg):

    # Commit changes
    _subCMD(ctx, arg)


# Create a main that is used by setup.cfg as console_script entry point
def main():
    templateXXX(obj={})

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()