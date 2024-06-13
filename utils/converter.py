import re
import numpy as np


traj_prefix = "["
traj_suffix = "]"
traj_separator = ", "
coord_prefix = "("
coord_suffix = ")"
coord_template = "{:.2f}"
coord_separator = ", "


def change_template(template: dict):
    r"""Change the template for trajectory and coordinate conversion.
    
    Args:
        template (dict): Template for trajectory and coordinate conversion.
    
    Examples::
        >>> template = {
            >>>     "traj_prefix": "[",
            >>>     "traj_suffix": "]",
            >>>     "traj_separator": ", ",
            >>>     "coord_prefix": "(",
            >>>     "coord_suffix": ")",
            >>>     "coord_template": "{:.2f}",
            >>>     "coord_separator": ", "
            >>> }
        >>> change_template(template)"""

    global traj_prefix, traj_suffix, traj_separator, coord_prefix, coord_suffix, coord_template, coord_separator
    if "traj_prefix" in template:
        traj_prefix = template["traj_prefix"]
    if "traj_suffix" in template:
        traj_suffix = template["traj_suffix"]
    if "traj_separator" in template:
        traj_separator = template["traj_separator"]
    if "coord_prefix" in template:
        coord_prefix = template["coord_prefix"]
    if "coord_suffix" in template:
        coord_suffix = template["coord_suffix"]
    if "coord_template" in template:
        coord_template = template["coord_template"]
    if "coord_separator" in template:
        coord_separator = template["coord_separator"]


def traj2text(trajectory: np.ndarray, prefix: str = "", suffix: str = ""):
    r"""Convert a trajectory to a string.
    
    Args:
        traj (np.ndarray): Trajectory to be converted, shape (frame, dim).
        prefix (str): Prefix of the converted string.
        suffix (str): Suffix of the converted string.
        
    Returns:
        str: Converted string.
        
    Examples::
        >>> traj = np.array([[1., 2.], [3., 4.]])
        >>> traj2text(traj)
        '[(1.00, 2.00), (3.00, 4.00)]'"""
    
    frame, dim = trajectory.shape

    coord_text_list = [coord_prefix + coord_separator.join([coord_template.format(trajectory[i, j]) for j in range(dim)]) + coord_suffix for i in range(frame)]
    text = traj_prefix + traj_separator.join(coord_text_list) + traj_suffix

    return prefix + text + suffix



def text2traj(description: str, frame: int = 12, dim: int = 2):
    r"""Convert a string to a trajectory.
    
    Args:
        description (str): String to be converted.
        frame (int): Number of frames in the trajectory.
        dim (int): Dimension of the trajectory.
        
    Returns:
        np.ndarray: Converted trajectory, shape (frame, dim).
        
    Examples::
        >>> description = '[(1.00, 2.00), (3.00, 4.00)]'
        >>> text2traj(description)
        array([[1., 2.],
               [3., 4.]])"""
    
    # Sanity check
    error = False
    if len(description) == 0:
        # Empty description
        error = True
    elif '[(' not in description or ')]' not in description:
        # Invalid format
        error = True
    
    # Try to parse the description
    if not error:
        try:
            description_cleanup = description[description.find('[('):description.find(')]') + 2]
            description_cleanup = re.sub('[^0-9()\[\],.\-\n]', '', description_cleanup.replace('(', '[').replace(')', ']'))
            description_list = eval(description_cleanup)
        except:
            error = True
            
    # validity check
    if not error and all(len(description_list[i]) == dim for i in range(len(description_list))):
        # wildcard for frame slightly different from the given frame
        if len(description_list) == frame - 1:
            description_list.append(description_list[-1])
        elif len(description_list) == frame + 1:
            description_list = description_list[:-1]
        
        # Convert to numpy array
        if not error and len(description_list) == frame:
            try:
                traj = np.array(description_list)
            except:
                error = True
        else:
            error = True
    else:
        error = True

    # Return the valid trajectory
    if not error:
        return traj
    else:
        return None


def batch_traj2txt(traj_list, prefix: str = "", suffix: str = ""):
    r"""Convert a list of trajectories to a list of strings.
    
    Args:
        traj_list (list or np.ndarray): List of trajectories to be converted, shape (batch, frame, dim).
        prefix (str): Prefix of the converted string.
        suffix (str): Suffix of the converted string.
        
    Returns:
        list: List of converted strings.
        
    Examples::
        >>> traj_list = [np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]])]
        >>> batch_traj2txt(traj_list)
        ['[(1.00, 2.00), (3.00, 4.00)]', '[(5.00, 6.00), (7.00, 8.00)]']"""
    
    return [traj2text(traj, prefix, suffix) for traj in traj_list]


def batch_text2traj(desc_list, frame: int = 12, dim: int = 2):
    r"""Convert a list of strings to a list of trajectories.
    
    Args:
        desc_list (list): List of strings to be converted.
        frame (int): Number of frames in the trajectory.
        dim (int): Dimension of the trajectory.
        
    Returns:
        list: List of converted trajectories, shape (batch, frame, dim).
    
    Examples::
        >>> desc_list = ['[(1.00, 2.00), (3.00, 4.00)]', '[(5.00, 6.00), (7.00, 8.00)]']
        >>> batch_text2traj(desc_list)
        [array([[1., 2.],
               [3., 4.]]), array([[5., 6.],
               [7., 8.]])]"""
    
    return [text2traj(desc, frame, dim) for desc in desc_list]


if __name__ == "__main__":
    traj = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    print(traj2text(traj))
    print(text2traj(traj2text(traj), frame=4, dim=2))
    print(batch_traj2txt([traj, traj]))
    print(batch_text2traj(batch_traj2txt([traj, traj]), frame=4, dim=2))
