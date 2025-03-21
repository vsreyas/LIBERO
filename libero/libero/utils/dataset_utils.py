import h5py
import numpy as np
import json


def get_dataset_info(dataset_path, filter_key=None, verbose=True):
    # extract demonstration list from file
    all_filter_keys = None
    f = h5py.File(dataset_path, "r")
    if filter_key is not None:
        # use the demonstrations from the filter key instead
        print("NOTE: using filter key {}".format(filter_key))
        demos = sorted(
            [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
        )
    else:
        # use all demonstrations
        demos = sorted(list(f["data"].keys()))

        # extract filter key information
        if "mask" in f:
            all_filter_keys = {}
            for fk in f["mask"]:
                fk_demos = sorted(
                    [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])]
                )
                all_filter_keys[fk] = fk_demos

    # put demonstration list in increasing episode order
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # extract length of each trajectory in the file
    traj_lengths = []
    action_min = np.inf
    action_max = -np.inf
    for ep in demos:
        traj_lengths.append(f["data/{}/actions".format(ep)].shape[0])
        action_min = min(action_min, np.min(f["data/{}/actions".format(ep)][()]))
        action_max = max(action_max, np.max(f["data/{}/actions".format(ep)][()]))
    traj_lengths = np.array(traj_lengths)

    problem_info = json.loads(f["data"].attrs["problem_info"])

    language_instruction = "".join(problem_info["language_instruction"])
    # report statistics on the data
    print("")
    print("total transitions: {}".format(np.sum(traj_lengths)))
    print("total trajectories: {}".format(traj_lengths.shape[0]))
    print("traj length mean: {}".format(np.mean(traj_lengths)))
    print("traj length std: {}".format(np.std(traj_lengths)))
    print("traj length min: {}".format(np.min(traj_lengths)))
    print("traj length max: {}".format(np.max(traj_lengths)))
    print("action min: {}".format(action_min))
    print("action max: {}".format(action_max))
    print("language instruction: {}".format(language_instruction.strip('"')))
    print("")
    print("==== Filter Keys ====")
    if all_filter_keys is not None:
        for fk in all_filter_keys:
            print("filter key {} with {} demos".format(fk, len(all_filter_keys[fk])))
    else:
        print("no filter keys")
    print("")
    if verbose:
        if all_filter_keys is not None:
            print("==== Filter Key Contents ====")
            for fk in all_filter_keys:
                print(
                    "filter_key {} with {} demos: {}".format(
                        fk, len(all_filter_keys[fk]), all_filter_keys[fk]
                    )
                )
        print("")
    env_meta = json.loads(f["data"].attrs["env_args"])
    print("==== Env Meta ====")
    print(json.dumps(env_meta, indent=4))
    print("")

    print("==== Dataset Structure ====")
    for ep in demos:
        print(
            "episode {} with {} transitions".format(
                ep, f["data/{}".format(ep)].attrs["num_samples"]
            )
        )
        for k in f["data/{}".format(ep)]:
            if k in ["obs", "next_obs"]:
                print("    key: {}".format(k))
                for obs_k in f["data/{}/{}".format(ep, k)]:
                    shape = f["data/{}/{}/{}".format(ep, k, obs_k)].shape
                    print(
                        "        observation key {} with shape {}".format(obs_k, shape)
                    )
            elif isinstance(f["data/{}/{}".format(ep, k)], h5py.Dataset):
                key_shape = f["data/{}/{}".format(ep, k)].shape
                print("    key: {} with shape {}".format(k, key_shape))

        if not verbose:
            break

    f.close()

def load_dataset_to_dict(dataset_path):
    """
    Loads the dataset from the given HDF5 file and returns it as a dictionary 
    organized by demonstration IDs.

    The structure of the returned dictionary is:
    
        data = {
            "demo_0": {
                "actions": np.array(...),
                "dones": np.array(...),
                "obs": {
                    "agentview_rgb": np.array(...),
                    "ee_ori": np.array(...),
                    ... (other observation keys)
                },
                "rewards": np.array(...),
                "robot_states": np.array(...),
                "states": np.array(...)
            },
            "demo_1": { ... },
            ...
        }

    Parameters:
        dataset_path (str): Path to the HDF5 dataset file.

    Returns:
        dict: A dictionary containing the dataset in a structure easy to use for training.
    """
    data_dict = {}
    with h5py.File(dataset_path, "r") as f:
        # Get demonstration IDs sorted by numerical order, assuming IDs are like "demo_0", "demo_1", etc.
        demos = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[-1]))
        for demo in demos:
            demo_group = f["data"][demo]
            demo_data = {}
            # Extract direct dataset keys
            for key in ["actions", "dones", "rewards", "robot_states", "states"]:
                if key in demo_group:
                    demo_data[key] = demo_group[key][()]
            # For observations, which are stored as a group
            if "obs" in demo_group:
                obs_group = demo_group["obs"]
                obs_data = {}
                for obs_key in obs_group.keys():
                    obs_data[obs_key] = obs_group[obs_key][()]
                demo_data["obs"] = obs_data
            data_dict[demo] = demo_data
    
    print("Dataset keys: ", data_dict.keys())
    print("Demo keys: ", data_dict["demo_1"].keys())
    print("Demo obs keys: ", data_dict["demo_1"]["obs"].keys())
    return data_dict