# -*- coding: utf-8 -*-
"""
Created: Nov 12, 2016
Updated: September 23, 2026
Author : Nicholas Pike
Email  : Npike@rad-con.com

Purpose: To calculate the Raman spectrum, at a user defined orientation and angle,
         by reading in data from an abinit calculation.  This program will read
         the anaddb output file "anaddb.out" or from a user specified file,
         extract the needed data, and output the result for plotting with your
         favorite program.

To run:  To run this script simply execute the following code
         python Raman_spec.py "name of input file"

Version: 0.0 - Initial building of program
         0.1 - Additional for angle dependent calculation
         0.2 - Additional user input required and number of output files reduced
         0.3 - Moved all user prompted input to an input file
         1.0 - Debugged, introduced data status and progress logging.

Input:  Input variables : filename, temp, laser_freq, spread,
        calctype, outname, relative_intensity, freq_unit,
        keep_file, n_freq, min_freq, max_freq.
        See a minimal input file at the end of the tutorial nlo.

Output: Program will output a text file containing the raman spectrum vs frequency
        and an outfile which outlines what happens in the calculation.

Bugs:   If you find a bug with this program or wish to see a feature added to
        this script please let me know at Npike@rad-con.com

        Happy Hunting!
"""
#begin main program

"""
Start of definations and other useful information
"""

import argparse
import linecache
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Optional, Any, Union

import numpy as np
from scipy import constants as sc

__version__ = '1.0'
_HARTREE_J = sc.physical_constants['Hartree energy'][0]
_EV_J = sc.physical_constants['electron volt-joule relationship'][0]
cm1_to_hartree = sc.speed_of_light * 100.0 * sc.h / _HARTREE_J
cm1_to_hz = sc.speed_of_light * 100.0
Hz_to_Ha = sc.h / _HARTREE_J
kb = sc.Boltzmann / _EV_J
hplank = sc.h / _EV_J
FINE_STRUCTURE_CONSTANT = sc.fine_structure
clight = 1.0 / FINE_STRUCTURE_CONSTANT  # Speed of light in atomic units (~137)
# Atomic unit of area (bohr^2) expressed in Angstrom^2, for reporting
# cross-sections computed in atomic units.
BOHR2_TO_ANG2 = (sc.physical_constants['Bohr radius'][0] * 1.0e10) ** 2
# Output file prefix. A module global because `run` rewrites it after
# check_repeat has resolved collisions and call_raman_menu reads it back when
# naming per-q-direction output files.
outname = 'raman.log'

# Tolerance constants for Raman calculations
RAMAN_TENSOR_TOLERANCE = 1e-10
RAMAN_NEGLIGIBLE_THRESHOLD = 1e-3
FREQUENCY_TOLERANCE = 1e-14

# Modes at or below this frequency are treated as acoustic and given no Raman
# intensity. anaddb reports an imaginary frequency as a negative number, so
# those fall below the cutoff too and are excluded by the same test.
ACOUSTIC_MODE_CUTOFF_CM1 = 1.0

# A q-direction whose Cartesian norm exceeds this is an LO direction; at or
# below it the block is the TO one. Separate from FREQUENCY_TOLERANCE, which
# this previously borrowed despite the two having unrelated units and scales.
Q_DIRECTION_TOLERANCE = 1e-14

# Channel order used for every 7-element intensity row, and for the columns of
# the intensity/spectrum output files. 'POWDER' is the orientation average and
# has no single pair of polarization vectors, so it is absent from
# POLARIZATION_MAP below and handled separately in raman_intensities.
POLARIZATION_ORDER = ('POWDER', 'XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ')

# Polarization vectors for Raman intensity calculations
POLARIZATION_MAP = {
    'XX': (np.array([1, 0, 0]), np.array([1, 0, 0])),
    'XY': (np.array([1, 0, 0]), np.array([0, 1, 0])),
    'XZ': (np.array([1, 0, 0]), np.array([0, 0, 1])),
    'YY': (np.array([0, 1, 0]), np.array([0, 1, 0])),
    'YZ': (np.array([0, 1, 0]), np.array([0, 0, 1])),
    'ZZ': (np.array([0, 0, 1]), np.array([0, 0, 1])),
}

# ==================== Logging System ====================

class LogLevel(Enum):
    """Logging severity levels for Raman calculations."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class RamanLogger:
    """Centralized logger for Raman spectrum calculations.
    
    Handles both console and file output with severity levels, progress tracking,
    and automatic file management.
    
    Example:
        >>> logger = RamanLogger('raman.log', level=LogLevel.INFO)
        >>> logger.info('Processing started')
        >>> logger.close()
    """
    
    def __init__(self, log_file: Optional[str] = None, level: LogLevel = LogLevel.INFO):
        """Initialize logger with optional file output.
        
        Args:
            log_file: Path to log file (None = console only)
            level: Minimum log level to display
        """
        self.log_file_path = log_file
        self.log_file = None
        self.level = level
        self._progress_enabled = True
        self._progress_last_pct = -1
        
        if log_file:
            self.open_log_file(log_file)
    
    def open_log_file(self, filename: str) -> None:
        """Open log file for writing.
        
        Args:
            filename: Path to log file
        """
        if self.log_file:
            self.log_file.close()
        self.log_file_path = filename
        # Explicit UTF-8: Python picks the locale encoding otherwise, which on
        # Windows is cp1252 and raises UnicodeEncodeError on any non-Latin-1
        # character written to the log. That was a hard crash mid-run, not a
        # mangled character, and it killed a real AlN run at the first mode.
        self.log_file = open(filename, 'w', encoding='utf-8')
    
    def close(self) -> None:
        """Close log file and flush output."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
    
    def set_level(self, level: LogLevel) -> None:
        """Set minimum logging level.
        
        Args:
            level: New log level
        """
        self.level = level
    
    def enable_progress(self, enabled: bool = True) -> None:
        """Enable or disable progress reporting.
        
        Args:
            enabled: Whether to show progress
        """
        self._progress_enabled = enabled
    
    def _write(self, message: str) -> None:
        """Write message to both console and file.

        The console write is guarded because sys.stdout uses the locale
        encoding, which on a default Windows console is cp1252 and raises
        UnicodeEncodeError on any character outside Latin-1. A decorative
        character in a status line should never abort a calculation, so an
        unencodable message is retried with the offending characters replaced
        rather than propagating. The log file is opened as UTF-8 and needs no
        such fallback.

        Args:
            message: Message to write
        """
        if self.log_file:
            self.log_file.write(f"{message}\n")
            self.log_file.flush()
        try:
            print(message)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
            print(message.encode(encoding, errors='replace').decode(encoding))
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        if self.level.value <= LogLevel.DEBUG.value:
            self._write(f"[DEBUG] {message}")
    
    def info(self, message: str) -> None:
        """Log info message."""
        if self.level.value <= LogLevel.INFO.value:
            self._write(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        if self.level.value <= LogLevel.WARNING.value:
            self._write(f"[WARNING] {message}")
    
    def error(self, message: str) -> None:
        """Log error message (always shown)."""
        self._write(f"[ERROR] {message}")
    
    def progress(self, current: int, total: int, label: str = "") -> None:
        """Report progress as percentage (~5% throttling).
        
        Args:
            current: Current position
            total: Total items
            label: Optional progress label
        """
        if not self._progress_enabled or total <= 0:
            return
        pct = int(100.0 * current / total)
        if pct >= self._progress_last_pct + 5 or pct == 100:
            self._progress_last_pct = pct
            label_str = f" {label}" if label else ""
            self.debug(f"Progress: {pct}%{label_str}")
    
    def write_spectrum_line(self, line: str, outfile: str) -> None:
        """Write a line to spectrum output file.
        
        Args:
            line: Data line to write
            outfile: Output file path
        """
        try:
            with open(outfile, 'a') as f:
                f.write(f"{line}\n")
        except IOError as e:
            self.error(f"Failed to write to {outfile}: {e}")


# Global logger instance
_logger: Optional[RamanLogger] = None


def get_logger() -> RamanLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = RamanLogger()
    return _logger


def set_logger(logger: RamanLogger) -> None:
    """Set the global logger instance.
    
    Args:
        logger: Logger to use globally
    """
    global _logger
    _logger = logger

# ==================== Configuration Dataclass ====================

@dataclass
class RamanConfig:
    """Configuration for Raman spectrum post-processing.

    `filename` and `laser_freq` have no defaults because neither has a
    meaningful one: there is nothing to read without an anaddb file, and a
    Raman cross-section is undefined without an incident photon energy.

    Attributes:
        filename: Path to the anaddb text output to read.
        laser_freq: Incident laser frequency as (value, unit_code). Must be > 0.
        temp: Temperature in Kelvin, used for the Bose-Einstein population factor.
        spread: Lorentzian half-width as (value, unit_code). Zero disables
            broadening, in which case per-mode intensities are written instead
            of a continuous spectrum.
        calctype: Calculation type code.
        outname: Output file name prefix.
        relative_intensity: Normalize each channel to its own maximum.
        freq_unit: Output frequency unit (0=Ha, 1=Hz, 2=cm^-1).
        keep_file: Overwrite an existing output file rather than writing to a
            numbered variant.
        n_freq: Number of frequency points in the broadened spectrum.
        min_freq: Lower bound as (value, unit_code); negative means auto-select.
        max_freq: Upper bound as (value, unit_code); negative means auto-select.
        verbose: Emit DEBUG-level output, including progress reporting.
    """
    filename: str
    laser_freq: Tuple[float, int]
    temp: float = 300.0
    spread: Tuple[float, int] = (0.0, 2)
    calctype: int = 0
    outname: str = 'raman'
    relative_intensity: bool = False
    freq_unit: int = 2
    keep_file: bool = False
    n_freq: int = 1000
    min_freq: Tuple[float, int] = (-1.0, 2)
    max_freq: Tuple[float, int] = (-1.0, 2)
    verbose: bool = False

    def validate(self) -> None:
        """Check the configuration describes a physically meaningful calculation.

        Raises:
            FileNotFoundError: If `filename` does not exist.
            ValueError: If the anaddb file is NetCDF rather than text, or if
                temp, laser_freq, spread or n_freq are outside their valid ranges.
        """
        if not self.filename:
            raise ValueError("Input is missing the required 'filename' field.")
        if str(self.filename).endswith('.nc'):
            raise ValueError('Raman_spec.py requires an anaddb text output, not NetCDF.')
        if not check_file(self.filename):
            raise FileNotFoundError(f"The anaddb output file '{self.filename}' was not found.")
        if self.temp <= 0:
            raise ValueError("'temp' must be greater than zero Kelvin.")
        if self.laser_freq[0] <= 0:
            raise ValueError("'laser_freq' must be greater than zero.")
        if self.spread[0] < 0:
            raise ValueError("'spread' cannot be negative.")
        if self.n_freq < 2:
            raise ValueError("'n_freq' must be at least 2.")

def _normalize_unit_string(unit_str: Union[str, Any]) -> str:
    """Normalize unit strings by lowercasing and replacing unicode variants.
    
    Args:
        unit_str: Unit string to normalize
        
    Returns:
        Normalized unit string
    """
    return str(unit_str).strip().lower().replace('^', '').replace('μ', 'u').replace('µ', 'u')

def get_unit(string: Union[str, Any]) -> int:
    """Convert a unit string to its numeric code.
    
    Args:
        string: Unit name ('ha'/'hartree', 'hz'/'hertz', 'cm-1', or wavelength like 'nm'/'um')
        
    Returns:
        Unit code: 0=Hartree, 1=Hz, 2=cm^-1
        
    Raises:
        ValueError: If unit is not recognized
    """
    normalized = _normalize_unit_string(string)
    if normalized in ('ha', 'hartree'):
        return 0
    if normalized in ('hz', 'hertz'):
        return 1
    if normalized in ('cm1', 'cm-1', 'wavenumber', 'nm', 'nanometer', 'nanometers', 'um', 'u m', 'micrometer', 'micrometers'):
        return 2
    raise ValueError(
        f"Unknown Raman frequency unit '{string}'. Use Ha, Hz, cm-1, or a wavelength like 532 nm."
    )


def _convert_wavelength_to_cm1(value: Union[float, str], unit: Union[str, Any]) -> float:
    """Convert a wavelength given in nm/um/cm to cm^-1 for Raman calculations.
    
    Args:
        value: Wavelength value (can be float or string)
        unit: Wavelength unit ('nm', 'um', 'cm', etc.)
        
    Returns:
        Equivalent frequency in cm^-1
    """
    normalized = _normalize_unit_string(unit)
    numeric = float(value)
    if normalized in ('nm', 'nanometer', 'nanometers'):
        return 1e7 / numeric
    if normalized in ('um', 'u m', 'micrometer', 'micrometers'):
        return 1e4 / numeric
    if normalized in ('cm', 'cm-1', 'wavenumber', 'centimeter', 'centimeters'):
        return numeric
    return numeric


def _normalize_raman_value(value: Union[float, str, List, Tuple], default_unit: int = 2) -> Tuple[float, int]:
    """Return a pair (numeric_value, unit_code) from a user value.
    
    Args:
        value: Input value (float, string "number unit", or [number, unit] list)
        default_unit: Default unit code if not specified (0=Ha, 1=Hz, 2=cm^-1)
        
    Returns:
        Tuple of (numeric_value, unit_code)
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        number, unit = value
        unit_code = get_unit(str(unit))
        normalized_unit = _normalize_unit_string(unit)
        if normalized_unit in ('nm', 'nanometer', 'nanometers', 'um', 'u m', 'micrometer', 'micrometers', 'cm', 'cm-1', 'wavenumber', 'centimeter', 'centimeters'):
            return _convert_wavelength_to_cm1(number, unit), unit_code
        return float(number), unit_code
    if isinstance(value, str):
        parts = value.split()
        if len(parts) == 2:
            number, unit = parts
            unit_code = get_unit(unit)
            normalized_unit = _normalize_unit_string(unit)
            if normalized_unit in ('nm', 'nanometer', 'nanometers', 'um', 'u m', 'micrometer', 'micrometers', 'cm', 'cm-1', 'wavenumber', 'centimeter', 'centimeters'):
                return _convert_wavelength_to_cm1(number, unit), unit_code
            return float(number), unit_code
        return float(parts[0]), default_unit
    return float(value), default_unit


def _field_value(fields: List[str], line: str, idx: int = 1) -> str:
    """Return field `idx` of a parsed input line.

    Args:
        fields: Whitespace-split tokens of the line.
        line: The original line, quoted in the error message.
        idx: Index of the value token to return.

    Returns:
        The requested token.

    Raises:
        ValueError: If the line has a key but no value at `idx`.
    """
    if len(fields) > idx:
        return fields[idx]
    raise ValueError(f"Line has key '{fields[0]}' but no value: {line.strip()}")


def read_input(user_filein: str) -> RamanConfig:
    """Read and validate a Raman spectrum input file.

    Boolean keys ('relative_intensity', 'keep_file', 'verbose') are flags:
    their presence alone turns them on. Value keys accept either "key number"
    or "key number unit", the latter allowing wavelengths such as "532 nm".

    Args:
        user_filein: Path to the input file.

    Returns:
        The parsed, validated configuration.

    Raises:
        FileNotFoundError: If the input file or the anaddb file it names is absent.
        ValueError: If a key is missing its value, 'laser_freq' was never given,
            or a value is outside its valid range.
    """
    if not check_file(user_filein):
        raise FileNotFoundError(f"Raman input file '{user_filein}' was not found.")

    settings: Dict[str, Any] = {}
    flag_keys = {'relative_intensity', 'keep_file', 'verbose'}
    value_parsers = {
        'filename': lambda f, l: _field_value(f, l),
        'temp': lambda f, l: float(_field_value(f, l)),
        'calctype': lambda f, l: int(_field_value(f, l)),
        'outname': lambda f, l: _field_value(f, l),
        'freq_unit': lambda f, l: get_unit(_field_value(f, l)),
        'n_freq': lambda f, l: int(_field_value(f, l)),
    }
    quantity_keys = {'laser_freq', 'spread', 'min_freq', 'max_freq'}

    with open(user_filein) as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            fields = stripped.split()
            key = fields[0]
            if key in flag_keys:
                settings[key] = True
            elif key in quantity_keys:
                raw = [fields[1], fields[2]] if len(fields) > 2 else _field_value(fields, line)
                settings[key] = _normalize_raman_value(raw)
            elif key in value_parsers:
                settings[key] = value_parsers[key](fields, line)

    if 'filename' not in settings:
        raise ValueError("Input is missing the required 'filename' field.")
    if 'laser_freq' not in settings:
        raise ValueError("Input is missing the required 'laser_freq' field.")

    config = RamanConfig(**settings)
    config.validate()

    global outname
    outname = check_repeat(config.outname, config.keep_file)
    config.outname = outname
    get_logger().info('Input file located and read in.\n')
    get_logger().info('The anaddb output file was found in the directory.')
    return config


def run(user_inputfile: Optional[str] = None, **kwargs: Any) -> RamanConfig:
    """Programmatic entry point.

    Accepts either a path to a Raman input file or explicit keyword arguments,
    for example: run(filename='anaddb.out', temp=300, laser_freq='563519657E3 Hz',
    spread='1E-5 Hz', freq_unit='cm1', ...) .

    Args:
        user_inputfile: Path to a Raman input file. If given, the only `kwargs`
            honoured is 'verbose', which can raise verbosity without editing
            the file; all other fields come from the file.
        **kwargs: RamanConfig fields. 'filename' and 'laser_freq' are required
            when no input file is given.

    Returns:
        The configuration the calculation ran with.

    Raises:
        ValueError: If a required field is missing or a value is out of range.
        FileNotFoundError: If the anaddb output file does not exist.
    """
    global outname
    get_logger().open_log_file(outname)
    try:
        if user_inputfile is not None:
            config = read_input(user_inputfile)
            if kwargs.get('verbose'):
                config.verbose = True
        else:
            converters = {
                'filename': str,
                'temp': float,
                'laser_freq': _normalize_raman_value,
                'spread': _normalize_raman_value,
                'calctype': int,
                'outname': str,
                'relative_intensity': bool,
                'freq_unit': lambda v: get_unit(str(v)),
                'keep_file': bool,
                'n_freq': int,
                'min_freq': _normalize_raman_value,
                'max_freq': _normalize_raman_value,
                'verbose': bool,
            }
            settings = {}
            for key, value in kwargs.items():
                normalized_key = key.strip().lower()
                if normalized_key in converters:
                    settings[normalized_key] = converters[normalized_key](value)
            if 'filename' not in settings:
                raise ValueError('A Raman input filename is required when no input file is given.')
            if 'laser_freq' not in settings:
                raise ValueError('A Raman laser_freq is required when no input file is given.')

            config = RamanConfig(**settings)
            config.validate()
            outname = check_repeat(config.outname, config.keep_file)
            config.outname = outname
            get_logger().info('Input file located and read in.\n')
            get_logger().info('The anaddb output file was found in the directory.')

        # Progress reporting is emitted at DEBUG level, so it stays silent
        # unless verbosity is raised here.
        if config.verbose:
            get_logger().set_level(LogLevel.DEBUG)

        print_header()
        deter_menu(config)
        return config
    finally:
        get_logger().close()

def check_file(filename: str) -> bool:
    """Check if a file exists in the current directory.
    
    Args:
        filename: Path to file to check
        
    Returns:
        True if file exists, False otherwise
    """
    if os.path.isfile(filename):
        logic  = True
    else:
        logic  = False

    return logic

def print_header() -> None:
    """Print a header to output file and terminal when starting program."""

    #Header for output file
    get_logger().info('')
    get_logger().info(f'++++++++++++++++++++++++ Version {__version__} ++++++++++++++++++++++++\n')
    get_logger().info('')
    get_logger().info('This program generates a Raman Spectrum after an anaddb '\
             'calculation.\n')
    get_logger().info('In this version, the program reads an the output file of an '\
             'anaddb\n calculation, finds the information it needs, and outputs '\
             'that\n information to a file.')
    get_logger().info('')
    get_logger().info('Author: Nicholas Pike')
    get_logger().info('Email:  Npike@rad-con.com')
    get_logger().info('')

    return

def check_repeat(filename: str, delete: bool) -> str:
    """Check if output file exists and handle naming accordingly.
    
    If delete is True and file exists, remove it. Otherwise create unique name
    by appending numeric suffix.
    
    Args:
        filename: Desired output file name
        delete: Whether to delete existing file with this name
        
    Returns:
        Filename to use (either original or with numeric suffix)
    """
    newfilename = filename
    if not check_file(filename):
        return newfilename
    if delete:
        os.remove(filename)
        return newfilename

    c = 1
    while check_file(newfilename):
        newfilename = f'{filename}{c}'
        c += 1
    return newfilename

def unit_to_ha(val: float, unit: int) -> float:
    """Convert a frequency value to Hartree units.

    Accepts a scalar or a NumPy array; arrays are converted elementwise.

    Args:
        val: Frequency value(s) in the unit given by `unit`.
        unit: 0=Hartree, 1=Hz, 2=cm^-1.

    Returns:
        Frequency in Hartree, same shape as `val`.

    Raises:
        ValueError: If `unit` is not 0, 1 or 2.
    """
    if unit == 0:   # val in Ha
        return val
    if unit == 1:   # val in Hz
        return val*Hz_to_Ha
    if unit == 2:   # val in cm1
        return val*cm1_to_hartree
    raise ValueError(f'Unknown frequency unit code {unit!r}; expected 0 (Ha), 1 (Hz) or 2 (cm-1).')

def ha_to_unit(val: float, unit: int) -> float:
    """Convert a frequency value from Hartree to target units.

    Accepts a scalar or a NumPy array; arrays are converted elementwise.

    Args:
        val: Frequency value(s) in Hartree.
        unit: Target unit (0=Hartree, 1=Hz, 2=cm^-1).

    Returns:
        Frequency in the target unit, same shape as `val`.

    Raises:
        ValueError: If `unit` is not 0, 1 or 2.
    """
    if unit == 0:   # val in Ha
        return val
    if unit == 1:   # val in Hz
        return val/Hz_to_Ha
    if unit == 2:   # val in cm1
        return val/cm1_to_hartree
    raise ValueError(f'Unknown frequency unit code {unit!r}; expected 0 (Ha), 1 (Hz) or 2 (cm-1).')

def _parse_raman_params(config: RamanConfig) -> Dict[str, Any]:
    """Convert a RamanConfig into the unit-resolved values the calculation uses.

    Frequencies arrive in whatever unit the user supplied. The laser frequency
    and broadening width are converted to Hartree, because that is the unit the
    mode energies and the Lorentzian are evaluated in. The spectrum bounds are
    instead converted into the requested OUTPUT unit, since they define the
    plotted axis rather than entering the physics.

    Args:
        config: Validated configuration.

    Returns:
        Dict with keys temp, laser, width, with_spreading, relative, freq_unit,
        delete, freqsteps, min_freq and max_freq.
    """
    T = config.temp
    laser, laser_unit = config.laser_freq
    width, width_unit = config.spread
    relative = config.relative_intensity
    freq_unit = config.freq_unit
    delete = config.keep_file
    freqsteps = config.n_freq
    min_freq, min_unit = config.min_freq
    max_freq, max_unit = config.max_freq

    # Determine if spreading (Lorentzian broadening) is enabled
    with_spreading = width > 0
    if with_spreading:
        width = unit_to_ha(width, width_unit)

    # Convert laser frequency to Hartree
    laser = unit_to_ha(laser, laser_unit)

    # Convert min/max frequencies: Ha → output freq_unit
    min_freq = unit_to_ha(min_freq, min_unit)
    min_freq = ha_to_unit(min_freq, freq_unit)
    max_freq = unit_to_ha(max_freq, max_unit)
    max_freq = ha_to_unit(max_freq, freq_unit)

    return {
        'temp': T,
        'laser': laser,
        'width': width,
        'with_spreading': with_spreading,
        'relative': relative,
        'freq_unit': freq_unit,
        'delete': delete,
        'freqsteps': freqsteps,
        'min_freq': min_freq,
        'max_freq': max_freq,
    }


def _validate_raman_data(modeenergy: List[Any], modedata: List[Any], qdirs: List[Any]) -> int:
    """Validate consistency of extracted Raman data.
    
    Args:
        modeenergy: List of mode energies by q-direction
        modedata: List of Raman tensors by q-direction
        qdirs: List of q-direction vectors
        
    Returns:
        Number of modes per q-direction
        
    Raises:
        ValueError: If data is invalid or inconsistent
    """
    if not modeenergy or not modedata or not qdirs:
        raise ValueError('No Raman modes or q-directions were parsed from anaddb.out.')
    if not (len(modeenergy) == len(modedata) == len(qdirs)):
        raise ValueError('Raman mode, tensor, and q-direction counts do not match.')
    
    n_modes = len(modeenergy[0])
    if n_modes == 0:
        raise ValueError('The Raman output contains q-directions but no modes.')
    if any(len(modes) != n_modes for modes in modeenergy):
        raise ValueError('Raman q-directions contain different numbers of modes.')
    if any(len(tensors) != n_modes for tensors in modedata):
        raise ValueError('Raman q-directions contain different numbers of tensors.')
    
    return n_modes


def _get_qdir_label(qdir: np.ndarray, outfile_base: str) -> Tuple[bool, str, str]:
    """Determine if q-direction is LO or TO, and generate output filename.
    
    Args:
        qdir: Q-direction vector
        outfile_base: Base filename for output
        
    Returns:
        Tuple of (is_LO, output_filename, label_string)
    """
    qdir_norm = np.sqrt(qdir[0]**2 + qdir[1]**2 + qdir[2]**2)
    is_LO = qdir_norm > Q_DIRECTION_TOLERANCE
    
    if is_LO:
        label = f'*** LO modes, in the q direction (cartesian coordinates) : {qdir[0]} {qdir[1]} {qdir[2]}'
        outfile = outfile_base + f'_{qdir[0]*1000:04.0f}_{qdir[1]*1000:04.0f}_{qdir[2]*1000:04.0f}'
    else:
        label = '*** TO modes'
        outfile = outfile_base
    
    return is_LO, outfile, label


def _print_mode_info(j: int, menergy: np.ndarray, rmatj: np.ndarray, max_rmatj: float, 
                     max_rarray: float, l_I: List[float], l_I_rel: List[float], 
                     laser_freq_cm1: float = 0.0, temperature_k: float = 300.0,
                     damping_fwhm_cm1: float = 0.0) -> None:
    """Print Raman tensor and intensity information for a single mode.
    
    Args:
        j: Mode index
        menergy: Array of mode energies
        rmatj: 3x3 Raman tensor for this mode
        max_rmatj: Maximum absolute value in this tensor
        max_rarray: Maximum absolute value across all tensors
        l_I: Absolute intensities (7 polarizations)
        l_I_rel: Relative intensities (7 polarizations)
        laser_freq_cm1: Laser frequency in cm^-1 (for cross-section calculation)
        temperature_k: Temperature in Kelvin, for the cross-section Bose factor
        damping_fwhm_cm1: Lorentzian FWHM in cm^-1; when > 0 a phonon lifetime
            and quality factor are reported for this mode
    """
    if max_rmatj < RAMAN_NEGLIGIBLE_THRESHOLD * max_rarray:
        get_logger().info(f' -- Mode {j+1}: {menergy[j][2]:10.2f} (cm-1) : Raman tensor is negligible')
    else:
        get_logger().info(f' -- Mode {j+1}: {menergy[j][2]:10.2f} (cm-1)')
        get_logger().info('    -- Raman tensor :')
        get_logger().info(f'    {rmatj[0][0]:17.9f} {rmatj[0][1]:17.9f} {rmatj[0][2]:17.9f}')
        get_logger().info(f'    {rmatj[1][0]:17.9f} {rmatj[1][1]:17.9f} {rmatj[1][2]:17.9f}')
        get_logger().info(f'    {rmatj[2][0]:17.9f} {rmatj[2][1]:17.9f} {rmatj[2][2]:17.9f}')
        
        # Calculate and display tensor analysis
        G0, G1, G2 = get_g012(rmatj)
        anisotropy = get_anisotropy_metric(G0, G1, G2)
        # Orientation-averaged ratio from this mode's own invariants. Was
        # get_depolarization_ratio(l_I[1], l_I[2]), the single-crystal
        # I_XY/I_XX, which has none of the bounds the label implies -- on real
        # AlN data that printed 1.47e9 for the 237.54 cm-1 E2 mode simply
        # because its tensor is purely off-diagonal and I_XX is numerical zero.
        depo_ratio = get_powder_depolarization(rmatj)
        
        # Calculate additional derived quantities
        raman_activity = get_raman_activity(rmatj)
        sym_analysis = decompose_tensor_symmetry(rmatj)
        
        get_logger().info('    -- Tensor Analysis :')
        get_logger().info(f'        G0 (isotropic) : {G0:12.5e} | G1 (asymmetric) : {G1:12.5e} | G2 (anisotropic) : {G2:12.5e}')
        
        # Build anisotropy line with conditional depolarization ratio
        anisotropy_line = f'        Anisotropy : {anisotropy * 100.0:6.2f}% '
        if depo_ratio is not None:
            anisotropy_line += f'| Depolarization Ratio (powder) : {depo_ratio:6.4f}'
        else:
            anisotropy_line += '| Depolarization Ratio (powder) : inf'
        get_logger().info(anisotropy_line)
        
        get_logger().info(f"        Raman Activity : {raman_activity:12.5e} | Optical Activity Ratio : {sym_analysis['optical_activity_ratio']:6.4f}")
        
        # Calculate theoretical Raman cross-section (DFT-based, no calibration needed)
        if laser_freq_cm1 > 0:
            sigma_theoretical = calculate_theoretical_raman_cross_section(
                rmatj, menergy[j][2], laser_freq_cm1, temperature_k)
            get_logger().info(f'        Theoretical Cross-Section (powder) : {sigma_theoretical:12.5e} Ang^2/sr')
        
        # Linewidth diagnostics, only when the user supplied a broadening to
        # interpret as this mode's physical damping width.
        if damping_fwhm_cm1 > 0:
            lw = estimate_linewidth_quality(menergy[j][2], damping_fwhm_cm1)
            get_logger().info(
                f"        Quality Factor Q : {lw['quality_factor']:10.2f} | "
                f"Phonon Lifetime : {lw['phonon_lifetime_ps']:10.4f} ps | "
                f"Linewidth : {lw['linewidth_quality']}")

        get_logger().info('    -- Raman intensities :')
        get_logger().info('        | powder     | (XX)      | (XY)      | (XZ)      | (YY)      | (YZ)      | (ZZ)')
        get_logger().info(f'    abs | {l_I[0]:10.5e}|{l_I[1]:10.5e}|{l_I[2]:10.5e}|{l_I[3]:10.5e}|{l_I[4]:10.5e}|{l_I[5]:10.5e}|{l_I[6]:10.5e}')
        get_logger().info(f'    rel | {l_I_rel[0]:11.2f}|{l_I_rel[1]:11.5f}|{l_I_rel[2]:11.5f}|{l_I_rel[3]:11.5f}|{l_I_rel[4]:11.5f}|{l_I_rel[5]:11.5f}|{l_I_rel[6]:11.5f}')
        get_logger().info('')


def _write_mode_intensities(j: int, menergy: np.ndarray, l_I: List[float], l_I_rel: List[float], 
                           relative: bool, freq_unit: int, outfile: str) -> None:
    """Write Raman intensities for a single mode to output file (no broadening).
    
    Args:
        j: Mode index
        menergy: Array of mode energies
        l_I: Absolute intensities (7 polarizations)
        l_I_rel: Relative intensities (7 polarizations)
        relative: Whether to write relative or absolute intensities
        freq_unit: Output frequency unit (0=Ha, 1=Hz, 2=cm^-1)
        outfile: Output file path
    """
    if relative:
        get_logger().write_spectrum_line(f' {menergy[j][freq_unit]:11.2f} {l_I_rel[0]:11.5f} {l_I_rel[1]:11.5f} {l_I_rel[2]:11.5f} {l_I_rel[3]:11.5f} {l_I_rel[4]:11.5f} {l_I_rel[5]:11.5f} {l_I_rel[6]:11.5f}', outfile)
    else:
        get_logger().write_spectrum_line(f' {menergy[j][freq_unit]:10.5e} {l_I[0]:10.5e} {l_I[1]:10.5e} {l_I[2]:10.5e} {l_I[3]:10.5e} {l_I[4]:10.5e} {l_I[5]:10.5e} {l_I[6]:10.5e}', outfile)


def _accumulate_spectrum(menergy: np.ndarray, rarray: List[np.ndarray], intensities: np.ndarray,
                        width: float, freq_unit: int, freqsteps: int,
                        min_freq: float, max_freq: float, relative: bool, outfile: str) -> None:
    """Calculate and write the Lorentzian-broadened Raman spectrum.

    Each mode contributes a Lorentzian of half-width `width` centred on its own
    energy, scaled by that mode's intensity in each of the seven polarization
    channels. Modes whose Raman tensor is entirely below
    RAMAN_TENSOR_TOLERANCE contribute nothing.

    A negative `min_freq` or `max_freq` means "choose automatically", in which
    case the range is taken from the mode frequencies themselves, padded by 5%.

    Args:
        menergy: Mode energies, shape (nmode, 3), as [Ha, Hz, cm^-1].
        rarray: One 3x3 Raman tensor per mode.
        intensities: Absolute intensities, shape (7, nmode), ordered
            [powder, XX, XY, XZ, YY, YZ, ZZ].
        width: Lorentzian half-width in Hartree.
        freq_unit: Output frequency unit (0=Ha, 1=Hz, 2=cm^-1).
        freqsteps: Number of frequency points to emit.
        min_freq: Lower bound of the spectrum in `freq_unit`, or negative to auto-select.
        max_freq: Upper bound of the spectrum in `freq_unit`, or negative to auto-select.
        relative: If True, normalize each channel to its own maximum.
        outfile: Path the spectrum rows are appended to.
    """
    n_modes = len(menergy)

    # Auto-range from the modes that actually contribute. The previous loop
    # updated its running bound only when that bound still exceeded the
    # current mode, so it settled on 0.95 times whichever mode happened to
    # come last rather than 0.95 times the smallest -- [100, 98] gave 95
    # instead of 93.1. It also scanned imaginary (negative) modes, which
    # carry no intensity but could drag the axis below zero and spend most of
    # the requested points on an empty region.
    all_freqs = np.asarray(menergy)[:, freq_unit] if n_modes else np.empty(0)
    physical = np.asarray(menergy)[:, 2] > ACOUSTIC_MODE_CUTOFF_CM1 if n_modes else np.empty(0, bool)
    usable = all_freqs[physical]
    if usable.size == 0:
        usable = all_freqs

    if min_freq >= 0.0:
        minf = min_freq
    elif usable.size:
        minf = float(usable.min()) * 0.95
    else:
        minf = 0.0

    if max_freq >= 0.0:
        maxf = max_freq
    elif usable.size:
        maxf = float(usable.max()) * 1.05
    else:
        maxf = 1.0

    frequencies = np.linspace(minf, maxf, freqsteps)

    # A mode's tensor magnitude does not depend on frequency, so the
    # active-mode mask is computed once rather than per frequency point.
    active = np.array([np.sum(np.abs(rmatj)) > RAMAN_TENSOR_TOLERANCE
                       for rmatj in rarray], dtype=bool)

    # Lorentzian weight of every mode at every frequency point, in one
    # broadcast instead of a freqsteps x nmode x 7 Python loop nest.
    freqs_ha = unit_to_ha(frequencies, freq_unit)
    mode_energies_ha = np.asarray(menergy)[:, 0]
    spread = width / ((freqs_ha[:, None] - mode_energies_ha[None, :]) ** 2 + width ** 2)
    spread[:, ~active] = 0.0

    # (nfreq, nmode) @ (nmode, 7) -> (nfreq, 7), transposed to (7, nfreq)
    ramanplot = (spread @ np.asarray(intensities).T).T

    if relative:
        maxval = ramanplot.max(axis=1)
        nonzero = maxval > 0.0
        ramanplot[nonzero] /= maxval[nonzero][:, None]

    get_logger().progress(freqsteps, freqsteps, 'spectrum')
    for i, freq in enumerate(frequencies):
        get_logger().write_spectrum_line('%10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e' %
                    (freq, ramanplot[0][i], ramanplot[1][i], ramanplot[2][i],
                     ramanplot[3][i], ramanplot[4][i], ramanplot[5][i], ramanplot[6][i]), outfile)


def _process_qdir(qdir: np.ndarray, menergy: np.ndarray, rarray: List[np.ndarray], 
                  params: Dict[str, Any], outfile_base: str, delete: bool, freq_unit: int) -> None:
    """Process one q-direction: calculate intensities and write output.
    
    Args:
        qdir: Q-direction vector
        menergy: Array of mode energies for this q-direction
        rarray: List of 3x3 Raman tensors for this q-direction
        params: Parsed Raman parameters from _parse_raman_params()
        outfile_base: Base filename for output
        delete: Whether to delete existing output files
        freq_unit: Output frequency unit
    """
    is_LO, outfile, label = _get_qdir_label(qdir, outfile_base)
    outfile = check_repeat(outfile, delete)
    
    get_logger().info('*' * 72)
    get_logger().info(label)
    get_logger().info('*' * 72)
    get_logger().info('')
    
    # Write header to output file
    if is_LO:
        get_logger().write_spectrum_line(f'#*** LO modes : {qdir[0]} {qdir[1]} {qdir[2]}', outfile)
    else:
        get_logger().write_spectrum_line('#*** TO modes', outfile)
    
    # Frequency unit label
    freq_labels = {0: '  freq (Ha)', 1: '  freq (Hz)', 2: ' freq (cm1)'}
    string_freq = freq_labels.get(freq_unit, ' freq (cm1)')
    
    if not params['relative']:
        get_logger().write_spectrum_line(f'#{string_freq} abs Ipowder   abs I(XX)   abs I(XY)   abs I(XZ)   abs I(YY)   abs I(YZ)   abs I(ZZ)', outfile)
    else:
        get_logger().write_spectrum_line(f'#{string_freq} rel Ipowder   rel I(XX)   rel I(XY)   rel I(XZ)   rel I(YY)   rel I(YZ)   rel I(ZZ)', outfile)
    
    n_modes = len(menergy)

    # Modes at or below ACOUSTIC_MODE_CUTOFF_CM1 get zero intensity in
    # raman_intensities. That discards the three acoustic modes, but a
    # NEGATIVE frequency is
    # anaddb's way of reporting an imaginary one, which means the structure is
    # dynamically unstable and the spectrum below is computed about a saddle
    # point rather than a minimum. Say so rather than silently dropping them.
    imaginary = [(k, menergy[k][2]) for k in range(n_modes) if menergy[k][2] < 0.0]
    if imaginary:
        get_logger().warning(
            f'{len(imaginary)} mode(s) have imaginary (negative) frequencies, '
            f'lowest {min(f for _, f in imaginary):.2f} cm-1: this structure is '
            f'dynamically UNSTABLE. They contribute nothing to the spectrum, and '
            f'the remaining intensities describe a non-equilibrium structure.')
        for k, freq in imaginary:
            get_logger().debug(f'    imaginary mode {k+1}: {freq:.2f} cm-1')

    # 'spread' is the Lorentzian HALF width used for broadening, so the full
    # width at half maximum that estimate_linewidth_quality expects is twice it.
    # Treating the user's broadening as the mode's physical damping width is a
    # modelling choice, not a computed anharmonic linewidth, so these numbers
    # are only as meaningful as the spread the user supplied.
    damping_fwhm_cm1 = (2.0 * params['width'] / cm1_to_hartree
                        if params['with_spreading'] else 0.0)

    get_logger().info('Calculating the raman intensities.')
    
    # Pre-compute every polarization once.
    abs_rows = []
    rel_rows = []
    for option in POLARIZATION_ORDER:
        abs_row, rel_row = raman_intensities(menergy, rarray, params['laser'],
                                             params['temp'], option)
        abs_rows.append(abs_row)
        rel_rows.append(rel_row)
    I_abs = np.array(abs_rows)   # (7, nmode)
    I_rel = np.array(rel_rows)   # (7, nmode)
    I_powder, I_xx, I_xy = I_abs[0], I_abs[1], I_abs[2]

    max_rarray = max((np.max(np.abs(tensor)) for tensor in rarray), default=0.0)
    
    # Process modes
    for j in range(n_modes):
        get_logger().progress(j, n_modes, f"mode {j+1}/{n_modes}")
        rmatj = rarray[j]
        max_rmatj = np.max(np.abs(rmatj))
        
        l_I = I_abs[:, j]
        l_I_rel = I_rel[:, j]

        # Convert laser frequency from Hartree to cm^-1 for display
        laser_freq_cm1 = params['laser'] / cm1_to_hartree if params['laser'] > 0 else 0.0
        _print_mode_info(j, menergy, rmatj, max_rmatj, max_rarray, l_I, l_I_rel,
                         laser_freq_cm1, params['temp'], damping_fwhm_cm1)
        
        if not params['with_spreading']:
            _write_mode_intensities(j, menergy, l_I, l_I_rel, params['relative'], freq_unit, outfile)
    
    # Display summary: Top modes ranked by intensity
    get_logger().info('')
    get_logger().info('*' * 72)
    get_logger().info('SUMMARY: Top 10 Strongest Modes (Powder Average)')
    get_logger().info('*' * 72)
    top_modes = rank_modes_by_intensity(menergy, I_powder, top_n=10)
    laser_freq_cm1 = params['laser'] / cm1_to_hartree if params['laser'] > 0 else 0.0
    
    strongest = max(I_powder) if len(I_powder) else 0.0
    for rank, (mode_idx, freq_cm1, intensity) in enumerate(top_modes, 1):
        # The orientation-averaged ratio, from the mode's own invariants.
        # This used to be get_depolarization_ratio(I_xx, I_xy), a single-
        # crystal ratio that carries none of the bounds reported alongside it
        # and reads 'N/A' whenever the XX channel happens to vanish -- which
        # it does for any mode whose tensor has no xx component, however
        # well-defined its powder ratio is.
        #
        # Suppressed entirely for a mode far below the strongest: anaddb
        # prints rounding noise rather than exact zeros for a Raman-inactive
        # mode, and a ratio built from it looks like a measurement. On real
        # AlN the silent 536.95 cm-1 mode reported 0.5240 from an intensity
        # nine orders of magnitude down.
        if strongest > 0.0 and intensity < RAMAN_NEGLIGIBLE_THRESHOLD * strongest:
            depo_str = "--"
        else:
            depo_ratio = get_powder_depolarization(rarray[mode_idx])
            depo_str = f"{depo_ratio:.4f}" if depo_ratio is not None else "inf"
        
        # Calculate theoretical cross-section
        sigma = calculate_theoretical_raman_cross_section(
            rarray[mode_idx], freq_cm1, laser_freq_cm1, params['temp']) if laser_freq_cm1 > 0 else 0.0
        sigma_str = f"{sigma:.3e}" if sigma > 0 else "N/A"
        
        get_logger().info(f'  {rank:2d}. Mode {mode_idx+1:3d} : {freq_cm1:10.2f} cm-1 | '
                         f'I = {intensity:10.5e} | sigma = {sigma_str} Ang^2/sr | rho = {depo_str}')
    
    # Display thermodynamic properties for top mode
    if len(top_modes) > 0:
        get_logger().info('')
        get_logger().info('*' * 72)
        get_logger().info('THERMODYNAMIC PROPERTIES (for strongest mode)')
        get_logger().info('*' * 72)
        strongest_idx = top_modes[0][0]
        strongest_freq = menergy[strongest_idx][2]
        T = params['temp']
        
        # Calculate per-mode contributions
        Cv = get_mode_heat_capacity(strongest_freq, T)
        S = get_mode_entropy(strongest_freq, T)
        
        get_logger().info(f'  Mode: {strongest_idx+1} | Frequency: {strongest_freq:.2f} cm-1 | T: {T:.1f} K')
        get_logger().info(f'  Heat Capacity Contribution: {Cv:.6f} kB (Boltzmann constant)')
        get_logger().info(f'  Entropy Contribution:       {S:.6f} kB')
        
        # Physical interpretation
        if Cv < 0.01:
            get_logger().info('  -> Mode is frozen (high frequency or low T)')
        elif Cv > 0.9:
            get_logger().info('  -> Mode is fully excited (low frequency or high T)')
        else:
            get_logger().info('  -> Mode is partially excited')
        get_logger().info('')
    
    # Generate spectrum with Lorentzian broadening if requested
    if params['with_spreading']:
        _accumulate_spectrum(menergy, rarray, I_abs, params['width'],
                           freq_unit, params['freqsteps'], params['min_freq'], params['max_freq'],
                           params['relative'], outfile)


def call_raman_menu(output: Tuple[Any, ...], keywords: List[Any], config: RamanConfig) -> None:
    """Main Raman spectrum calculation and output routine.

    Processes phonon data, calculates Raman intensities for all polarizations,
    and writes results to output files.

    Args:
        output: Tuple with parsed anaddb data (from load_anaddb)
        keywords: List with calculation metadata (Abinit version, etc.)
        config: Validated Raman configuration
    """
    # Parse and convert parameters
    params = _parse_raman_params(config)
    
    # Extract data
    modeenergy = output[1]
    modedata = output[2]
    qdirs = output[4]
    
    # Count modes from first q-direction
    n_modes = len(modeenergy[0]) if modeenergy else 0

    #remainder of printing goes to log file only
    get_logger().info('')
    get_logger().info(f'Anaddb created with Abinit Version: {keywords[0]}')
    get_logger().info('')

    # Print header of output file
    if not params['with_spreading']:
        outfile_base = f'{outname}_intensity'
    else:
        outfile_base = f'{outname}_spec'

    # Start loop on q directions
    for idir, qdir in enumerate(qdirs):
        get_logger().progress(idir, len(qdirs), f"q-direction {idir+1}/{len(qdirs)}")
        
        # Prepare data for this q-direction
        menergy = get_modeenergy(modeenergy[idir])
        rarray = get_ramanmatrix(modedata[idir])
        
        # Process this q-direction
        _process_qdir(qdir, menergy, rarray, params, outfile_base, params['delete'], params['freq_unit'])

    return

def load_anaddb(infile: str) -> Tuple[List[Any], Tuple[Any, ...]]:
    """Load ABINIT anaddb output file and extract Raman data.
    
    Reads anaddb output file twice: first to find keywords/metadata, then to
    extract phonon and Raman tensor data.
    
    Args:
        infile: Path to anaddb output file
        
    Returns:
        Tuple of (keywords_list, output_tuple) where output_tuple contains
        extracted phonon and Raman tensor data
    """
    keywords =['', #Abinit version
               False, #dieflag - frequency dependent dielectric constant
               0,     #nfreq   - number of frequency steps (int, 0 = not found)
               False, #nlflag  - Raman tensor and nonlinear optical tensor
               False, #elaflag - elastic tensor flag
               False, #piezoflag - Piezoelectric tensor
               ]
    #First, read the output file and determine if any of the keywords are present
    with open(infile,'r') as f:
        for num,line in enumerate(f,1):
            if line.startswith( '.Version'):
                l = line.strip('\n').split()
                keywords[0] = str(l[1])
            elif line.startswith( '     dieflag'):
                l = line.strip('\n').split()
                dieflag = int(l[len(l)-1])
                if dieflag > 0 and dieflag != 2:
                    keywords[1] = True
                else:
                    keywords[1] = False
            elif line.startswith( '       nfreq'):
                l = line.strip('\n').split()
                if len(l) > 0:
                    keywords[2] = int(l[-1])
            elif line.startswith( '      nlflag'):
                l = line.strip('\n').split()
                if len(l) > 0 and int(l[-1]) > 0:
                    keywords[3] = True
                else:
                    keywords[3] = False

                #Now that keywords are read in and stored. We can read the file again and
                #figure out what information needs to be stored


    # Storage slots, indexed by the callers of this function:
    #   [0] unused, reserved
    #   [1] modeenergy   [2] modedata   [3] diedata   [4] qdirs
    # Slot 0 has never been assigned; it is kept so the existing positional
    # indices stay valid rather than renumbering every caller.
    outinfo= [None,0,0,0,0]
    qdirs = []
    modeenergy = []
    modedata = []
    diedata = []

    if keywords[1] and not keywords[2]:
        # dieflag asked for a frequency-dependent dielectric tensor but no
        # nfreq echo was found, so there is no row count to allocate. The old
        # code left nfreq as False and built np.zeros(shape=(False, 7)) -- a
        # valid but empty (0, 7) array -- then reported a successful
        # extraction having written nothing. Say so instead.
        get_logger().warning(
            'dieflag requests a frequency-dependent dielectric tensor but no '
            "'nfreq' value was found in the file, so the number of frequency "
            'points is unknown. Skipping dielectric extraction.')
        keywords[1] = False

    if keywords[1]: #dielectric tensor as a function of frequency
        #Do something
        get_logger().info('Starting extraction of the dielectric Tensor as a function of frequency.')
        diedata=np.zeros(shape=(keywords[2],7))
        with open(infile,'r') as f:
            for num,line in enumerate(f,1):
                if line.startswith(' Frequency(Hartree)    Dielectric constant                Reflectivity'):
                    c=2
                    for c in range(2,keywords[2]+2):
                        data = linecache.getline(infile,num+c)
                        l=data.strip('\n').split()
                        diedata[c-2][0]= l[0]
                        diedata[c-2][1]= l[1]
                        diedata[c-2][2]= l[2]
                        diedata[c-2][3]= l[3]
                        diedata[c-2][4]= l[4]
                        diedata[c-2][5]= l[5]
                        diedata[c-2][6]= l[6]

        outinfo[3] = diedata
        get_logger().info('Finished extraction of the dielectric Tensor as a function of frequency.')

    if keywords[3]: #Raman tensor and modes
        get_logger().info('Starting extraction of the raman tensor.')
        with open(infile,'r') as f:
            lines = f.readlines()
            iqdir = 0
            for num, line in enumerate(lines, 1):
                if 'raman susceptibilit' in line.lower():
                    modeenergy.append([])
                    modedata.append([])
                    iqdir += 1
                    qdir = None
                    lower_line = line.lower()
                    if 'transverse' in lower_line: # TO mode
                        qdir = np.array([0.0,0.0,0.0])
                    elif 'non-analyticity' in lower_line: # LO mode
                        fields = lines[num].strip().split() if num < len(lines) else []
                        if len(fields) < 6:
                            actual_line = lines[num].strip() if num < len(lines) else "<end of file>"
                            raise ValueError(
                                f"Could not parse Raman q-direction at line {num}. "
                                f"Expected format: 'Non-analyticity direction : X Y Z' "
                                f"but got: '{actual_line}'"
                            )
                        try:
                            qdir = np.array([float(fields[ii + 3]) for ii in range(3)])
                        except ValueError as error:
                            actual_line = lines[num].strip() if num < len(lines) else "<end of file>"
                            raise ValueError(
                                f"Invalid Raman q-direction at line {num}. "
                                f"Expected three numeric values for [X, Y, Z] coordinates "
                                f"but got: '{actual_line}'. Values at positions 4-6: {fields[3:6] if len(fields) > 5 else 'missing'}"
                            ) from error
                    else:
                        raise ValueError(
                            f"Unrecognized Raman q-direction block at line {num}. "
                            f"Expected 'transverse' or 'non-analyticity' in line, "
                            f"but got: '{line.strip()}'"
                        )
                    qdirs.append(qdir)
                if line.strip().startswith('Mod') and 'cm-1)' in line:
                    if iqdir == 0:
                        raise ValueError(
                            f"Raman mode found before a susceptibility block at line {num}. "
                            f"Expected a 'raman susceptibilit' or 'non-analyticity' block first, "
                            f"but got: '{line.strip()}'"
                        )
                    fields = line.split('(', 1)
                    try:
                        onemode_energy = float(fields[1].split()[0])
                    except (IndexError, ValueError) as error:
                        raise ValueError(
                            f"Invalid Raman mode frequency at line {num}. "
                            f"Expected format: 'Mode ... (NUMBER cm-1)' "
                            f"but got: '{line.strip()}'"
                        ) from error
                    onemode_raman = np.zeros([3,3])
                    for ii in range(3):
                        fields = lines[num + ii].strip().split() if num + ii < len(lines) else []
                        if len(fields) < 4:
                            actual_line = lines[num + ii].strip() if num + ii < len(lines) else "<end of file>"
                            raise ValueError(
                                f"Incomplete Raman tensor at line {num + ii}. "
                                f"Expected format: 'component_X component_Y component_Z ...' with at least 3 values, "
                                f"but got: '{actual_line}' (only {len(fields)} fields found)"
                            )
                        try:
                            onemode_raman[ii] = [float(value) for value in fields[1:4]]
                        except ValueError as error:
                            actual_line = lines[num + ii].strip() if num + ii < len(lines) else "<end of file>"
                            raise ValueError(
                                f"Invalid Raman tensor values at line {num + ii}. "
                                f"Expected three numeric values for tensor row, "
                                f"but got: '{actual_line}'. Values at positions 2-4: {fields[1:4] if len(fields) >= 4 else 'missing'}"
                            ) from error
                    modeenergy[iqdir-1].append(onemode_energy)
                    modedata[iqdir-1].append(onemode_raman)

        if not qdirs or not any(modeenergy):
            raise ValueError(
                f"No Raman modes were found in '{infile}'. "
                f"File must contain Raman tensor data marked by 'raman susceptibilit' blocks. "
                f"Check that ABINIT anaddb was run with nlflag > 0 (nonlinear optical tensor calculation)."
            )
        if any(qdir is None for qdir in qdirs):
            raise ValueError(
                f"A Raman q-direction could not be parsed in '{infile}'. "
                f"All 'raman susceptibilit' blocks must have associated q-direction data. "
                f"Check for 'non-analyticity direction' or 'transverse' markers."
            )

        outinfo[1] = modeenergy
        outinfo[2] = modedata
        outinfo[4] = qdirs
        
        # Validate data consistency before returning
        _validate_raman_data(modeenergy, modedata, qdirs)

        get_logger().info('Finished extraction of the raman tensor.')
        get_logger().info('')

    return keywords,outinfo

def get_modeenergy(modeenergy: List[Union[float, str]]) -> np.ndarray:
    """Convert phonon mode energies to multiple frequency units.
    
    Args:
        modeenergy: List of mode energies in cm^-1
        
    Returns:
        Numpy array of shape (nmode, 3) with energies in [Ha, Hz, cm^-1]
    """
    cm1 = np.asarray(modeenergy, dtype=float)
    return np.column_stack((cm1 * cm1_to_hartree, cm1 * cm1_to_hz, cm1))

def get_ramanmatrix(modedata: List[np.ndarray]) -> List[np.ndarray]:
    """Zero out Raman tensor elements below RAMAN_TENSOR_TOLERANCE.

    Elements that small are numerical noise from the anaddb finite-difference
    derivatives rather than physical polarizability, and leaving them in
    perturbs the tensor invariants used for the depolarization ratio.

    Args:
        modedata: One 3x3 Raman tensor per mode.

    Returns:
        A new list of 3x3 arrays with sub-tolerance elements set to zero. The
        input arrays are not modified.
    """
    cleaned = []
    for tensor in modedata:
        tensor = np.array(tensor, dtype=float)
        tensor[np.abs(tensor) < RAMAN_TENSOR_TOLERANCE] = 0.0
        cleaned.append(tensor)
    return cleaned

def get_g012(m: np.ndarray) -> Tuple[float, float, float]:
    """Rotational invariants of a Raman tensor, in ABINIT's convention.

    Splitting the tensor into its symmetric part s = (m + m^T)/2 and its
    antisymmetric part a = (m - m^T)/2, the three invariants are

        G0 = (1/3) * Tr(m)^2                  isotropic
        G1 = sum_ij a_ij^2                    antisymmetric
        G2 = sum_ij s_ij^2 - (1/3) * Tr(m)^2  symmetric anisotropic

    These match the 'Spherical averages : G0= G1= G2=' line anaddb prints
    under each mode's tensor, which is what makes the values here directly
    comparable with the source anaddb.out. Verified against a real
    AlN wurtzite run: all seven Raman-active modes agree with anaddb to the
    precision of its printed tensors.

    Two earlier departures from that convention are corrected here. G2 carried
    a spurious factor of 1/2, making it exactly half of anaddb's value; and it
    was built from Tr(m.m) rather than from the symmetric part alone, which for
    a non-symmetric tensor adds Tr(a.a) < 0 and so subtracts the antisymmetric
    weight from an invariant that must not depend on it. Both mattered: the
    powder intensity in `raman_intensities` combines these as 10*G0 + 4*G2 and
    5*G1 + 3*G2, coefficients that reproduce the standard orientation-averaged
    Placzek result (I_par ~ 45a^2 + 4*gamma^2, I_perp ~ 3*gamma^2, with
    gamma^2 = (3/2)*G2) ONLY in this convention. With a half-scale G2 the
    isotropic and anisotropic contributions were weighted against each other
    two to one.

    G2 is non-negative by Cauchy-Schwarz, since Tr(s)^2 <= 3*sum_ij s_ij^2;
    the clamp below only absorbs floating-point noise near zero.

    Args:
        m: 3x3 Raman tensor.

    Returns:
        Tuple (G0, G1, G2), all >= 0.
    """
    m = np.asarray(m, dtype=float)
    sym = 0.5 * (m + m.T)
    antisym = 0.5 * (m - m.T)

    trace = float(np.trace(sym))
    isotropic = (1.0 / 3.0) * trace * trace

    G0 = isotropic
    G1 = float(np.sum(antisym * antisym))
    G2 = max(0.0, float(np.sum(sym * sym)) - isotropic)

    return G0, G1, G2


def get_powder_depolarization(tensor: np.ndarray) -> Optional[float]:
    """Depolarization ratio of one mode, from its tensor invariants.

    The orientation-averaged (powder) ratio for 90 degree scattering is

        rho = I_perp / I_par = (5*G1 + 3*G2) / (10*G0 + 4*G2)

    using the same invariants and the same coefficients `raman_intensities`
    uses for its POWDER channel, so the two agree by construction.

    This is the quantity the familiar bounds describe: rho = 0 for a purely
    isotropic tensor, rho = 3/4 for a symmetric traceless one, and rho above
    3/4 only once the tensor has an antisymmetric part. A single-crystal ratio
    such as I_XY / I_XX is a different observable with none of those bounds,
    and must not be substituted for it.

    Args:
        tensor: 3x3 Raman tensor.

    Returns:
        The ratio, or None when I_par vanishes -- a purely antisymmetric mode,
        where the ratio genuinely diverges rather than being ill-defined.
    """
    if np.sum(np.abs(tensor)) <= RAMAN_TENSOR_TOLERANCE:
        # A Raman-inactive mode. anaddb still prints a tensor of rounding
        # noise for these, and a ratio formed from it is noise divided by
        # noise -- on real AlN data the silent 536.95 cm-1 mode produced a
        # confident-looking 0.5240 from elements of order 1e-8.
        return None

    G0, G1, G2 = get_g012(tensor)
    i_par = 10.0 * G0 + 4.0 * G2
    i_perp = 5.0 * G1 + 3.0 * G2
    if i_par <= 0.0:
        return None
    return i_perp / i_par

# ==================== Derived Quantity Calculations ====================

def get_depolarization_ratio(I_parallel: float, I_perpendicular: float) -> Optional[float]:
    """Ratio of two measured intensities, I_perp / I_par.

    A general-purpose helper for any pair of channels. For the POWDER
    depolarization ratio -- the one carrying the familiar 0 to 3/4 bounds --
    use `get_powder_depolarization`, which works from the tensor invariants
    directly and cannot be handed a mismatched pair of channels.

    No upper bound is imposed. A large ratio is a real result for a mode whose
    parallel channel is weak, not evidence of a tensor-extraction failure, and
    an earlier version discarded any ratio above 100 as 'physically
    impossible' while logging an error that had no basis.

    Args:
        I_parallel: Intensity in the parallel channel.
        I_perpendicular: Intensity in the perpendicular channel.

    Returns:
        The ratio, or None if the parallel intensity is not positive (the
        ratio diverges and has no finite value to report).
    """
    if I_parallel <= 0:
        return None
    return I_perpendicular / I_parallel

def get_anisotropy_metric(G0: float, G1: float, G2: float) -> float:
    """Calculate anisotropy metric from Raman tensor invariants.
    
    Anisotropy = G2 / (G0 + G2) measures the degree of tensor anisotropy.
    - 0.0: Completely isotropic tensor (only G0 non-zero)
    - 1.0: Completely anisotropic tensor (only G2 non-zero)
    - 0.5: Mixed isotropic and anisotropic character

    Also measures departure from spherical symmetry. High anisotropy
    makes modes easier to identify in spectra.

    G1 is accepted for a uniform (G0, G1, G2) call signature but is
    deliberately NOT part of the ratio: antisymmetric scattering is a separate
    channel rather than a degree of anisotropy within the symmetric one. Use
    `decompose_tensor_symmetry`'s 'optical_activity_ratio' to quantify it.

    Args:
        G0: Isotropic component
        G1: Antisymmetric component (unused, see above)
        G2: Anisotropic (symmetric-traceless) component

    Returns:
        Anisotropy metric (0.0 to 1.0)
    """
    total = G0 + G2
    if total <= 0:
        return 0.0
    return G2 / total

def rank_modes_by_intensity(menergy: np.ndarray, intensities: np.ndarray, 
                           top_n: Optional[int] = None) -> List[Tuple[int, float, float]]:
    """Rank modes by Raman intensity and return top N.
    
    Useful for identifying the strongest modes in a spectrum, which are
    typically the most observable in experiments.
    
    Args:
        menergy: Array of mode energies (nmode, 3) with [Ha, Hz, cm-1]
        intensities: Array of intensities (nmode)
        top_n: Number of top modes to return (None = all, sorted)
        
    Returns:
        List of tuples: (mode_index, frequency_cm1, intensity)
        sorted by intensity in descending order
    """
    # Create list of (index, frequency, intensity)
    mode_list = [(i, menergy[i][2], intensities[i]) for i in range(len(menergy))]
    
    # Sort by intensity (descending)
    mode_list.sort(key=lambda x: x[2], reverse=True)
    
    if top_n is not None:
        mode_list = mode_list[:top_n]
    
    return mode_list

def get_raman_activity(tensor: np.ndarray) -> float:
    """Calculate Raman activity from tensor elements.
    
    Raman activity A = Σᵢⱼ (αᵢⱼ)² is a simplified metric for the strength
    of Raman scattering. Unlike intensity (which depends on laser frequency
    and temperature), activity is an intrinsic tensor property.
    
    For a fully isotropic tensor: A = (1/3)(Tr α)²
    For general tensors: activity increases with tensor element magnitudes.
    
    Args:
        tensor: 3x3 Raman polarizability tensor
        
    Returns:
        Raman activity (sum of squared tensor elements)
    """
    return float(np.sum(tensor**2))

def decompose_tensor_symmetry(tensor: np.ndarray) -> Dict[str, Any]:
    """Decompose Raman tensor into symmetric and antisymmetric parts.
    
    Symmetric part αₛᵢⱼ = (αᵢⱼ + αⱼᵢ)/2 represents normal Raman scattering.
    Antisymmetric part αₐᵢⱼ = (αᵢⱼ - αⱼᵢ)/2 represents optical activity and
    circular dichroism (relevant for chiral crystals).
    
    The ratio of antisymmetric to total activity indicates how much the
    mode exhibits optical activity effects.
    
    Args:
        tensor: 3x3 Raman polarizability tensor
        
    Returns:
        Dict with:
        - 'symmetric': Symmetric part (3x3)
        - 'antisymmetric': Antisymmetric part (3x3)
        - 'symmetric_activity': Sum of squared symmetric elements
        - 'antisymmetric_activity': Sum of squared antisymmetric elements
        - 'optical_activity_ratio': Ratio of antisym to total activity (0-1)
    """
    symmetric = (tensor + tensor.T) / 2.0
    antisymmetric = (tensor - tensor.T) / 2.0
    
    sym_activity = float(np.sum(symmetric**2))
    antisym_activity = float(np.sum(antisymmetric**2))
    total_activity = sym_activity + antisym_activity
    
    optical_activity_ratio = 0.0
    if total_activity > RAMAN_TENSOR_TOLERANCE:
        optical_activity_ratio = antisym_activity / total_activity
    
    return {
        'symmetric': symmetric,
        'antisymmetric': antisymmetric,
        'symmetric_activity': sym_activity,
        'antisymmetric_activity': antisym_activity,
        'optical_activity_ratio': optical_activity_ratio,
    }

def get_mode_heat_capacity(mode_frequency_cm1: float, T: float) -> float:
    """Calculate per-mode contribution to heat capacity (at constant volume).

    This is the single harmonic oscillator (Einstein) result, evaluated for
    one mode at its own frequency:
    Cᵥ = kB * (ℏω/kB*T)² * exp(ℏω/kB*T) / [exp(ℏω/kB*T) - 1]²

    Not the Debye model, which this docstring previously named: that one
    integrates a specific density of states over many modes and takes a Debye
    temperature, neither of which appears here.
    
    Physical interpretation:
    - At high T (T >> ℏω/kB): Cᵥ → kB (classical Dulong-Petit limit)
    - At low T (T << ℏω/kB): Cᵥ → 0 (mode is "frozen out")
    - Peak around T ~ ℏω/kB: Mode is actively excited
    
    Args:
        mode_frequency_cm1: Mode frequency in cm^-1
        T: Temperature in Kelvin
        
    Returns:
        Heat capacity contribution in units of Boltzmann constant (kB)
    """
    if T <= 0:
        return 0.0
    
    # Convert cm-1 to eV using Planck's constant
    # E [eV] = h [eV·s] * c [cm/s] * ν [cm-1] / c [cm/s] = h*c*ν
    # Use hplank = h/eV_J and cm1_to_hz for conversion
    mode_freq_hz = mode_frequency_cm1 * cm1_to_hz  # Convert to Hz
    mode_energy_ev = hplank * mode_freq_hz  # Energy in eV
    
    if mode_energy_ev <= 0:
        return 0.0
    
    # Calculate dimensionless parameter x = ℏω / kB*T
    x = mode_energy_ev / (kb * T)
    
    if x < 1e-10:
        # High temperature limit: Cᵥ ≈ kB
        return 1.0
    elif x > 100:
        # Low temperature limit: Cᵥ ≈ 0
        return 0.0
    else:
        # General case: use full formula
        exp_x = np.exp(x)
        return x**2 * exp_x / (exp_x - 1.0)**2

def get_mode_entropy(mode_frequency_cm1: float, T: float) -> float:
    """Calculate per-mode contribution to entropy (at constant volume).

    The single harmonic oscillator (Einstein) result for one mode, not the
    Debye model this docstring previously named:
    S = kB * [ℏω/(kB*T) * 1/(exp(ℏω/kB*T)-1) - ln(1 - exp(-ℏω/kB*T))]
    
    Physical interpretation:
    - At high T: S → kB * ln(kB*T/ℏω) (logarithmically increases)
    - At low T: S → 0 (frozen phonon mode, no entropy)
    - Related to phonon population by n = 1/(exp(ℏω/kB*T) - 1)
    
    Args:
        mode_frequency_cm1: Mode frequency in cm^-1
        T: Temperature in Kelvin
        
    Returns:
        Entropy contribution in units of Boltzmann constant (kB)
    """
    if T <= 0:
        return 0.0
    
    mode_freq_hz = mode_frequency_cm1 * cm1_to_hz
    mode_energy_ev = hplank * mode_freq_hz
    
    if mode_energy_ev <= 0:
        return 0.0
    
    x = mode_energy_ev / (kb * T)
    
    if x < 1e-10:
        # High temperature limit: S ≈ kB * (1 + ln(kB*T/ℏω))
        return 1.0 + np.log(kb * T / mode_energy_ev)
    elif x > 100:
        # Low temperature limit: S ≈ 0
        return 0.0
    else:
        # General case: use numerically stable form
        # S = x/(exp(x)-1) - ln(1-exp(-x))
        exp_x = np.exp(x)
        exp_minus_x = 1.0 / exp_x  
        arg = 1.0 - exp_minus_x  
        return x / (exp_x - 1.0) - np.log(arg)

def estimate_linewidth_quality(mode_frequency_cm1: float, damping_width_cm1: float) -> Dict[str, float]:
    """Estimate phonon lifetime and quality factor from a Lorentzian linewidth.

    The quality factor Q = w / g compares the mode frequency to its full width
    at half maximum. Because both are in the same unit the ratio is
    unit-independent, so cm^-1 is used throughout.

    - Q > 100: long-lived phonon, well-ordered crystal
    - Q ~ 10-100: moderate damping, some disorder
    - Q < 10: heavily damped, amorphous or defect-rich

    The lifetime follows from the Lorentzian convention used elsewhere in this
    module: a spectral FWHM of g (in ordinary frequency) corresponds to
    tau = 1 / (2*pi*g). Equivalently the FWHM in ANGULAR frequency is 1/tau.

    Args:
        mode_frequency_cm1: Mode frequency in cm^-1.
        damping_width_cm1: Lorentzian FWHM in cm^-1. Note the `spread` input
            variable is a HALF width, so callers must double it.

    Returns:
        Dict with 'quality_factor' (dimensionless), 'phonon_lifetime_ps'
        (picoseconds) and 'linewidth_quality' ('high'/'moderate'/'low', or
        'unknown' when either input is non-positive).
    """
    if mode_frequency_cm1 <= 0 or damping_width_cm1 <= 0:
        return {
            'quality_factor': float('inf'),
            'phonon_lifetime_ps': float('inf'),
            'linewidth_quality': 'unknown'
        }

    Q = mode_frequency_cm1 / damping_width_cm1

    # tau = 1 / (2*pi*g) with g the FWHM in ordinary frequency (Hz).
    damping_freq_hz = damping_width_cm1 * cm1_to_hz
    lifetime_ps = 1.0e12 / (2.0 * np.pi * damping_freq_hz)

    if Q > 100:
        quality = 'high'
    elif Q > 10:
        quality = 'moderate'
    else:
        quality = 'low'

    return {
        'quality_factor': float(Q),
        'phonon_lifetime_ps': float(lifetime_ps),
        'linewidth_quality': quality,
    }

def calculate_theoretical_raman_cross_section(raman_tensor: np.ndarray,
                                             mode_frequency_cm1: float,
                                             laser_frequency_cm1: float,
                                             temperature_k: float = 300.0) -> float:
    """Differential Stokes Raman cross-section of one mode, from theory alone.

    Evaluates the orientation-averaged (powder) differential cross-section

        dsigma/dOmega = 2*pi * (omega_s^4 / c^4) * (n(omega_m,T) + 1)
                        / (2*omega_m) * (10*G0 + 5*G1 + 7*G2)

    entirely in Hartree atomic units, then converts the resulting area from
    bohr^2 to Angstrom^2. omega_s is the scattered (Stokes) photon frequency,
    n is the Bose-Einstein population of the mode, and G0/G1/G2 are the
    rotational invariants of the Raman tensor from `get_g012`.

    This is the same expression `raman_intensities` evaluates for its 'POWDER'
    channel, so a mode's cross-section here and its powder intensity there sit
    on one consistent scale rather than two unrelated ones.

    No empirical prefactor, scaling constant or reference-material calibration
    is applied. The absolute magnitude therefore carries the approximations of
    the underlying DFT Raman tensor (and of the dimensionless-polarizability
    convention anaddb writes it in), and should be treated as a first-principles
    estimate rather than a measured cross-section.

    Args:
        raman_tensor: 3x3 Raman polarizability tensor from anaddb.
        mode_frequency_cm1: Raman mode frequency in cm^-1.
        laser_frequency_cm1: Incident laser frequency in cm^-1.
        temperature_k: Temperature in Kelvin for the Bose-Einstein factor.

    Returns:
        Differential cross-section in Angstrom^2 per steradian, or 0.0 if the
        mode is non-Raman-active, has non-positive frequency, or lies above the
        laser energy (no Stokes channel).
    """
    if mode_frequency_cm1 <= 0 or laser_frequency_cm1 <= 0 or temperature_k <= 0:
        return 0.0

    # Stokes scattered photon: incident energy less the energy left in the crystal.
    scattered_freq_cm1 = laser_frequency_cm1 - mode_frequency_cm1
    if scattered_freq_cm1 <= 0:
        return 0.0

    G0, G1, G2 = get_g012(raman_tensor)
    invariant = 10.0 * G0 + 5.0 * G1 + 7.0 * G2
    if invariant <= RAMAN_TENSOR_TOLERANCE:
        return 0.0

    mode_ha = mode_frequency_cm1 * cm1_to_hartree
    scattered_ha = scattered_freq_cm1 * cm1_to_hartree

    # Bose-Einstein occupation of the mode at this temperature.
    bose = 1.0 / (np.exp(hplank * mode_frequency_cm1 * cm1_to_hz / (kb * temperature_k)) - 1.0)

    sigma_atomic = (2.0 * np.pi * scattered_ha ** 4 / clight ** 4
                    * (bose + 1.0) / (2.0 * mode_ha) * invariant)

    return float(sigma_atomic * BOHR2_TO_ANG2)

def raman_intensities(menergy: np.ndarray, rarray: List[np.ndarray], laser: float, T: float, option: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Raman intensity for a given polarization and mode set.
    
    Calculates both absolute and relative intensities using Bose-Einstein
    distribution and Raman tensor decomposition.
    
    Args:
        menergy: Array of mode energies with shape (nmode, 3) containing [Ha, Hz, cm^-1]
        rarray: List of 3x3 Raman tensor matrices (one per mode)
        laser: Laser frequency in Hartree
        T: Temperature in Kelvin
        option: Polarization type ('POWDER', 'XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ')
        
    Returns:
        Tuple of (absolute_intensities, relative_intensities) arrays
        
    Raises:
        ValueError: If T <= 0, laser <= 0, or menergy/rarray length mismatch
    """
    if T <= 0:
        raise ValueError('Raman temperature must be greater than zero Kelvin.')
    if laser <= 0:
        raise ValueError('Raman laser frequency must be greater than zero.')
    if len(menergy) != len(rarray):
        raise ValueError('Raman energies and tensors must have the same length.')
    if option != 'POWDER' and option not in POLARIZATION_MAP:
        raise ValueError(f"Unknown polarization option: {option}")

    nmode = len(menergy)
    Intensities     = np.zeros(nmode)
    Intensities_rel = np.zeros(nmode)
    for j in range(nmode):
        mode_cm1 = menergy[j][2]
        if mode_cm1 <= ACOUSTIC_MODE_CUTOFF_CM1:
            # Acoustic modes, and the negative frequencies anaddb writes for
            # imaginary ones, carry no Stokes intensity. _process_qdir warns
            # separately about the imaginary case.
            continue

        mode_ha = menergy[j][0]
        scattered_ha = laser - mode_ha
        if scattered_ha <= 0.0:
            # The mode lies at or above the incident photon energy, so there
            # is no Stokes-scattered photon. Previously this evaluated
            # (mode - laser)**4 regardless and reported a large intensity,
            # disagreeing with calculate_theoretical_raman_cross_section,
            # which returns 0.0 for exactly this case.
            continue

        mbose = 1.0/(np.exp(hplank*menergy[j][1]/(kb*T))-1.0)
        pref = 1.0/(2.0*mode_ha)*scattered_ha**4/clight**4*(mbose+1.0)
        if option == 'POWDER':
            G0,G1,G2 = get_g012(rarray[j])
            # The 2*pi belongs to the orientational average and so applies to
            # this channel only; the fixed-polarization channels below are
            # |e_s . alpha . e_i|^2 at one orientation, where no solid-angle
            # integration has been performed. The two therefore sit on
            # different scales by that factor even though they are written as
            # neighbouring columns -- see this module's README.
            pref  = (2*np.pi)*pref
            Ipar  = pref*(10.0*G0+4.0*G2)
            Iperp = pref*( 5.0*G1+3.0*G2)
            Intensities[j] = Ipar + Iperp
        else:
            Inpol, Outpol = POLARIZATION_MAP[option]
            modearray = np.asarray(rarray[j], dtype=float)
            alpha_ij = np.dot(Outpol,np.dot(modearray,Inpol))
            Intensities[j] = alpha_ij**2*pref

    # Guarded rather than max(Intensities): an empty mode list is a valid
    # (if degenerate) input and used to raise ValueError from the builtin.
    if nmode:
        max_int = float(np.max(Intensities))
        if max_int > 0.0:
            Intensities_rel = Intensities/max_int

    return Intensities,Intensities_rel

def deter_menu(config: RamanConfig) -> None:
    """Determine which calculations to perform based on anaddb output flags.

    Reads the anaddb file, detects which calculations were performed,
    and runs appropriate post-processing (Raman, dielectric, etc.).

    Args:
        config: Validated Raman configuration
    """
    infile = config.filename

    keywords,output = load_anaddb(infile)

    if keywords[1]:
        # run module to print Dielectric tensor as a function of frequency
        outfile = f'{outname}_dielectric_freq'
        get_logger().write_spectrum_line('# Freq    dielectric      Reflectivity',outfile)
        get_logger().write_spectrum_line('#           x y z x y z  ',outfile)
        for j in range(int(keywords[2])):
            o3j=output[3][j]
            get_logger().write_spectrum_line(f'{o3j[0]:e} {o3j[1]:e} {o3j[2]:e} {o3j[3]:e} {o3j[4]:e} {o3j[5]:e} {o3j[6]:e}', outfile)
    if keywords[3]:
        # Choose to identify the modes characteristics or make a Raman Spectrum
        get_logger().info('Entering Raman Spectrum Calculation.')
        get_logger().info('')
        call_raman_menu(output, keywords, config) #Calls the menu for the Raman calculation

    get_logger().info('Thank you for using this program.')

    return

def main(argv: Optional[List[str]] = None) -> RamanConfig:
    """CLI entry point. Accepts either a single input-file argument or direct flags.

    Args:
        argv: Command line arguments (None uses sys.argv[1:])

    Returns:
        The configuration the calculation ran with.
    """
    if argv is None:
        argv = sys.argv[1:]
    if any('SPYDER' in name for name in os.environ):
        user_inputfile = 'input_raman'
        return run(user_inputfile)

    if not argv:
        raise SystemExit(
            'Usage: python Raman_spec.py <raman_input_file> [--verbose]\n'
            '       python Raman_spec.py --filename anaddb.out --temp 300 --laser_freq "563519657E6 Hz" ...'
        )

    if argv[0].startswith('-'):
        parser = argparse.ArgumentParser(description='Raman spectrum post-processor.')
        parser.add_argument('--filename', required=True)
        parser.add_argument('--temp', type=float, default=300.0)
        parser.add_argument('--laser_freq', required=True)
        parser.add_argument('--spread', default='1E-5 Hz')
        parser.add_argument('--calctype', type=int, default=0)
        parser.add_argument('--outname', default='raman')
        parser.add_argument('--relative_intensity', action='store_true')
        parser.add_argument('--freq_unit', default='cm1')
        parser.add_argument('--keep_file', action='store_true')
        parser.add_argument('--n_freq', type=int, default=1000)
        parser.add_argument('--min_freq', default='0 cm1')
        parser.add_argument('--max_freq', default='2000 cm1')
        parser.add_argument('--verbose', action='store_true',
                            help='Emit DEBUG output, including progress reporting.')
        args = parser.parse_args(argv)
        return run(
            filename=args.filename,
            temp=args.temp,
            laser_freq=args.laser_freq,
            spread=args.spread,
            calctype=args.calctype,
            outname=args.outname,
            relative_intensity=args.relative_intensity,
            freq_unit=args.freq_unit,
            keep_file=args.keep_file,
            n_freq=args.n_freq,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            verbose=args.verbose,
        )

    # Input-file form. --verbose is accepted alongside the file so verbosity can
    # be raised without editing the input; the file's own 'verbose' flag also works.
    return run(argv[0], verbose=any(a in ('--verbose', '-v') for a in argv[1:]))


# Program runs via this if statement (needed for windows computers):
if __name__ == '__main__':
    main()

#Ends program