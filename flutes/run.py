import subprocess
import tempfile
from typing import Dict, List, NamedTuple, Optional, TypeVar, Union

from .log import log
from .types import PathType

__all__ = [
    "run_command",
    "CommandResult",
    "error_wrapper",
]


class CommandResult(NamedTuple):
    command: Union[str, List[str]]
    r"""The executed command in its original form."""
    return_code: int
    r"""The return code of the executed command."""
    captured_output: Optional[bytes]
    r"""The terminal output of the command. ``captured_output`` will be ``None`` unless an exception occurred, or
    ``return_output`` is set to ``True``."""


ExcType = TypeVar('ExcType', bound=Exception)


def error_wrapper(err: ExcType) -> ExcType:
    r"""Wrap exceptions raised in :py:mod:`subprocess` to output captured output by default.
    """
    if not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        return err

    def __str__(self):
        string = super(self.__class__, self).__str__()
        if self.output:
            try:
                output = self.output.decode('utf-8')
            except UnicodeEncodeError:  # ignore output
                string += "\nFailed to parse output."
            else:
                string += "\nCaptured output:\n" + '\n'.join([f'    {line}' for line in output.split('\n')])
        else:
            string += "\nNo output was generated."
        return string

    # Dynamically create a new type that overrides __str__, because replacing __str__ on instances don't work.
    err_type = type(err)
    new_type = type(err_type.__name__, (err_type,), {"__str__": __str__})

    err.__class__ = new_type
    return err  # type: ignore[return-value]


MAX_OUTPUT_LENGTH = 8192


def run_command(args: Union[str, List[str]], *,
                env: Optional[Dict[str, str]] = None, cwd: Optional[PathType] = None, timeout: Optional[float] = None,
                verbose: bool = False, return_output: bool = False, ignore_errors: bool = False,
                **kwargs) -> CommandResult:
    r"""A wrapper over ``subprocess.check_output`` that prevents deadlock caused by the combination of pipes and
    timeout. Output is redirected into a temporary file and returned only on exceptions or when return code is nonzero.

    In case an OSError occurs, the function will retry for a maximum for 5 times with exponential back-off. If error
    still occurs, we just re-raise it.

    :param args: The command to run. Should be either a `str` or a list of `str` depending on whether ``shell`` is True.
    :param env: Environment variables to set before running the command. Defaults to None.
    :param cwd: The working directory of the command to run. If None, uses the default (probably user home).
    :param timeout: Maximum running time for the command. If running time exceeds the specified limit,
        ``subprocess.TimeoutExpired`` is thrown.
    :param verbose: If ``True``, print out the executed command and output.
    :param return_output: If ``True``, the captured output is returned. Otherwise, the return code is returned.
    :param ignore_errors: If ``True``, exceptions will not be raised. A special return code of -32768 indicates a
        ``subprocess.TimeoutExpired`` error.
    :return: An instance of :class:`CommandResult`.
    """
    cwd_str = str(cwd) if cwd is not None else None
    if verbose:
        log((cwd_str or "") + "> " + repr(args), timestamp=False, include_proc_id=False)
    with tempfile.TemporaryFile() as f:
        try:
            ret = subprocess.run(args, check=True, stdout=f, stderr=subprocess.STDOUT,
                                 timeout=timeout, env=env, cwd=cwd_str, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            f.seek(0)
            output = f.read()
            if len(output) > MAX_OUTPUT_LENGTH:  # truncate if longer than 8192 characters
                output = b"*** (previous output truncated) ***\n" + output[-MAX_OUTPUT_LENGTH:]
            if ignore_errors:
                return_code = e.returncode if isinstance(e, subprocess.CalledProcessError) else -32768
                return CommandResult(args, return_code, output)
            else:
                e.output = output
                raise error_wrapper(e) from None
        if return_output or ret.returncode != 0 or verbose:
            f.seek(0)
            output = f.read()
            if verbose:
                try:
                    log(output.decode('utf-8'), timestamp=False, include_proc_id=False)
                except UnicodeDecodeError:
                    for line in output.split(b"\n"):
                        log(str(line), timestamp=False, include_proc_id=False)
            return CommandResult(args, ret.returncode, output)
    return CommandResult(args, ret.returncode, None)
