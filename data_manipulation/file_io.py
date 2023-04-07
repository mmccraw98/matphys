import numpy as np
import pandas as pd
import os
import pickle
import igor

def next_path(path_pattern):
    """
    https://stackoverflow.com/questions/17984809/how-do-i-create-an-incrementing-filename-in-python
    Finds the next free path in an sequentially named list of files
    e.g. path_pattern = 'file-%s.txt':
    file-1.txt
    file-2.txt
    file-3.txt
    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b

def get_files(directory, req_ext=None):
    '''
    gets all the files in the given directory
    :param directory: str directory from which you want to load files from
    :param req_ext: optional str required tc_data extension
    :return: list of str names of the files in the given directory
    '''
    if req_ext is None:
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    else:
        return [os.path.join(directory, f) for f in os.listdir(directory) if
                os.path.isfile(os.path.join(directory, f)) and req_ext in f]
        

def get_folders(directory):
    '''
    gets all the folders in the given directory
    :param directory: str directory from which you want the sub-directories
    :return: list of str names of the sub-directories
    '''
    return [f.path for f in os.scandir(directory) if f.is_dir()]

# appropriating some functions from from https://github.com/N-Parsons/ibw-extractor
def from_repr(s):
    """Get an int or float from its representation as a string"""
    # Strip any outside whitespace
    s = s.strip()
    # "NaN" and "inf" can be converted to floats, but we don't want this
    # because it breaks in Mathematica!
    if s[1:].isalpha():  # [1:] removes any sign
        rep = s
    else:
        try:
            rep = int(s)
        except ValueError:
            try:
                rep = float(s)
            except ValueError:
                rep = s
    return rep


def fill_blanks(lst):
    """Convert a list (or tuple) to a 2 element tuple"""
    try:
        return (lst[0], from_repr(lst[1]))
    except IndexError:
        return (lst[0], "")


def flatten(lst):
    """Completely flatten an arbitrarily-deep list"""
    return list(_flatten(lst))


def _flatten(lst):
    """Generator for flattening arbitrarily-deep lists"""
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        elif item not in (None, "", b''):
            yield item


def process_notes(notes):
    """Splits a byte string into an dict"""
    # Decode to UTF-8, split at carriage-return, and strip whitespace
    note_list = list(map(str.strip, notes.decode(errors='ignore').split("\r")))
    note_dict = dict(map(fill_blanks, [p.split(":") for p in note_list]))

    # Remove the empty string key if it exists
    try:
        del note_dict[""]
    except KeyError:
        pass
    return note_dict


def ibw2dict(filename):
    """Extract the contents of an *ibw to a dict"""
    data = igor.binarywave.load(filename)
    wave = data['wave']

    # Get the labels and tidy them up into a list
    labels = list(map(bytes.decode,
                      flatten(wave['labels'])))

    # Get the notes and process them into a dict
    notes = process_notes(wave['note'])

    # Get the data numpy array and convert to a simple list
    wData = np.nan_to_num(wave['wData']).tolist()

    # Get the filename from the file - warn if it differs
    fname = wave['wave_header']['bname'].decode()
    input_fname = os.path.splitext(os.path.basename(filename))[0]
    if input_fname != fname:
        print("Warning: stored filename differs from input file name")
        print("Input filename: {}".format(input_fname))
        print("Stored filename: {}".format(str(fname) + " (.ibw)"))

    return {"filename": fname, "labels": labels, "notes": notes, "data": wData}


def ibw2df(filename):
    data = ibw2dict(filename)
    headers = data['labels']
    return pd.DataFrame(data['data'], columns=headers)


def load(file_path):
    """
    Load data from a file using either ibw loader numpy, pandas, or pickle.

    Parameters
    ----------
    file_path : str
        The name or path of the file to load.

    Returns
    -------
    data : numpy.ndarray or pandas.DataFrame or object
        The data loaded from the file.

    Raises
    ------
    ValueError
        If the file extension is not recognized or if there is an error loading the data.
    """
    try:
        ext = os.path.splitext(file_path)[1]

        if ext in ('.xlsx', '.csv', '.txt'):
            # Load data from an Excel or CSV file using pandas
            if ext == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            # Drop any index columns
            df.reset_index(drop=True, inplace=True)

            # Estimate headers if not available
            if df.columns.duplicated().any():
                df = pd.read_csv(file_path, header=None)
                df.columns = [f"Column{i}" for i in range(1, len(df.columns) + 1)]

            # Return the data as a DataFrame
            return df
        
        elif ext == '.ibw':
           # Load data from an ibw file using ibw2df
           return ibw2df(file_path)

        elif ext == '.pkl':
            # Load data from a pickle file using pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data

    except Exception as e:
        raise ValueError(f"Error loading data from file '{file_path}': {e}")


def save(data, file_path, file_type, overwrite=False):
    """
    Save data to a file in a specified format.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame or object
        The data to save.
    file_path : str
        The file path to save the data to.
    file_type : str
        The file type to use for saving the data.
    overwrite : bool, optional
        Whether to allow overwriting existing files (default is False).

    Raises
    ------
    ValueError
        If the file type is not recognized, if the data cannot be saved in the specified format,
        or if there is an error saving the data.
    """
    try:
        ext = os.path.splitext(file_path)[1]

        # If the file already exists and overwrite is False, get the next free file path
        if os.path.exists(file_path) and not overwrite:
            file_path = next_path(file_path)

        if file_type == 'pkl':
            # Save data as a pickle file using pickle
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

        elif file_type in ('csv', 'xlsx', 'txt'):
            # Save data as a CSV, Excel, or text file using pandas
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            if file_type == 'csv':
                data.to_csv(file_path, index=False)
            elif file_type == 'xlsx':
                data.to_excel(file_path, index=False)
            elif file_type == 'txt':
                data.to_csv(file_path, index=False, sep='\t')

        else:
            # Raise an error if the file type is not recognized
            raise ValueError(f"Unrecognized file type '{file_type}'")

    except TypeError as e:
        raise ValueError(f"Error saving data to file '{file_path}': {e}. Please check that the data can be saved in the specified format.")
    except Exception as e:
        raise ValueError(f"Error saving data to file '{file_path}': {e}")
    
