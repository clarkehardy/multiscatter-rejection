import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import scipy.integrate as si
from joblib import Parallel, delayed

def get_pmt_positions(x, y, rows=100, n_per_row=100, diam=15., max_rad=100.):
    """Get the coordinates of all PMTs in an array

    :param x: x coordinate of the central PMT in mm
    :type x: float
    :param y: y coordinate of the central PMT in mm
    :type y: float
    :param rows: number of rows of PMTs, defaults to 24
    :type rows: int, optional
    :param n_per_row: number of PMTs per row, defaults to 24
    :type n_per_row: int, optional
    :param diam: PMT diameter in mm, defaults to 15.
    :type diam: float, optional
    :param max_rad: radius of the detector containing the PMTs in mm, defaults to 100.
    :type max_rad: float, optional
    :return: array containing the x and y coordinates of all PMTs
    :rtype: np.ndarray
    """
    center_row = rows // 2 + rows % 2 - 1
    center_n = n_per_row // 2 + n_per_row % 2 - 1
    x0 = x - center_row*diam*np.sqrt(3)/2.
    y0 = y - center_n*diam - (center_row % 2)*diam/2.
    xy_pos = []
    for i in range(rows):
        for j in range(n_per_row):
            x_center = x0 + i*diam*np.sqrt(3)/2.
            y_center = y0 + j*diam + (i%2)*diam/2.
            if np.sqrt(x_center**2 + y_center**2) + diam/2. > max_rad:
                continue
            xy_pos.append(np.array((x_center, y_center)))
    return np.array(xy_pos)

def lrf(r, A, r0, a, b, alpha):
    rho = r/r0
    return A*np.exp(-a*rho/(1 + rho**(1 - alpha)) - b/(1 + rho**(-alpha)))

def get_ss_events(num_events, max_sep):
    points_rt = np.random.uniform([0., 0.], [(max_sep/2)**2, 2.*np.pi], size=(num_events, 2))
    points_rt[:, 0] = np.sqrt(points_rt[:, 0])
    points_xy = np.zeros_like(points_rt)
    points_xy[:, 0] = points_rt[:, 0] * np.cos(points_rt[:, 1])
    points_xy[:, 1] = points_rt[:, 0] * np.sin(points_rt[:, 1])
    return points_xy

def independent_random_sampling(d, R):
    return 2*d/R**2*((2./np.pi)*np.arccos(d/(2.*R)) - d/(np.pi*R)*np.sqrt(1 - (d/(2.*R))**2))

def uniform_separation(d, R):
    return np.ones(len(d))/(2.*R)

def exp_separation(sep_array, max_sep):
    mfp = max_sep/3. # max separation is three mean free paths
    unscaled = np.exp(-sep_array/mfp)
    return unscaled/np.trapezoid(unscaled, sep_array)

def get_ms_events(n_events, max_sep, sep_pdf):
    angle = np.random.uniform(0, 2.*np.pi, n_events)
    site1 = np.array((-max_sep*np.cos(angle)/2., -max_sep*np.sin(angle)/2.)) \
            + max_sep*(np.random.rand(2*n_events).reshape((2, n_events)) - 0.5)/2.
    sep_array = np.linspace(0, max_sep, 1000)
    sep_cdf = si.cumulative_trapezoid(sep_pdf(sep_array, max_sep/2.), sep_array, initial=0)
    sep_dist = np.interp(np.random.rand(n_events), sep_cdf, sep_array)
    site2 = site1 + np.array((sep_dist*np.cos(angle), sep_dist*np.sin(angle)))
    return site1.T, site2.T

def plot_ss_event(event_sites, event_counts, event_ind, pmt_pos_array, pmt_diam, tpc_rad):
    fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')

    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('lightgray')
    norm = mcolors.LogNorm(vmin=1, vmax=np.amax(event_counts))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    ax.set_xlim([-1.2*tpc_rad, 1.2*tpc_rad])
    ax.set_ylim([-1.2*tpc_rad, 1.2*tpc_rad])
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ [mm]')
    ax.set_ylabel('$y$ [mm]')
    ax.add_artist(Circle((0, 0), tpc_rad*1.02, fc='none', ec='k', lw=2))

    for pos, counts in zip(pmt_pos_array, event_counts[event_ind,:]):
        ax.add_artist(Circle(pos, pmt_diam/2., fc=sm.to_rgba(counts)))
    ax.plot(*event_sites[event_ind], marker='.', ls='none', color='limegreen', label=' = event position')

    ax.legend(loc='lower right', handletextpad=0, handlelength=1., borderpad=0, frameon=False, fontsize=12)
    fig.colorbar(sm, ax=ax, label='Number of detected photons', shrink=0.8)
    return fig, ax

def plot_ms_event(event_site1, event_site2, event_counts, event_ind, pmt_pos_array, pmt_diam, tpc_rad, show_labels=False):
    fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')

    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('lightgray')
    norm = mcolors.LogNorm(vmin=1, vmax=np.amax(event_counts))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    ax.set_xlim([-1.2*tpc_rad, 1.2*tpc_rad])
    ax.set_ylim([-1.2*tpc_rad, 1.2*tpc_rad])
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ [mm]')
    ax.set_ylabel('$y$ [mm]')
    ax.add_artist(Circle((0, 0), tpc_rad*1.02, fc='none', ec='k', lw=2))

    i = 0
    for pos, counts in zip(pmt_pos_array, event_counts[event_ind,:]):
        ax.add_artist(Circle(pos, pmt_diam/2., fc=sm.to_rgba(counts)))
        if show_labels:
            ax.text(*pos, str(i), ha='center', va='center', fontsize=6, color='r')
        i += 1
    ax.plot(*event_site1[event_ind], marker='.', ls='none', color='cyan', label=' = event positions')
    ax.plot(*event_site2[event_ind], marker='.', ls='none', color='cyan')

    ax.legend(loc='lower right', handletextpad=0, handlelength=1., borderpad=0, frameon=False, fontsize=12)
    fig.colorbar(sm, ax=ax, label='Number of detected photons', shrink=0.8)
    return fig, ax

def plot_selected_pmts(counts, pmt_indices, event_id, pmt_pos_array, pmt_diam, tpc_rad):
    fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')

    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('lightgray')
    norm = mcolors.LogNorm(vmin=1, vmax=np.amax(counts))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    ax.set_xlim([-1.2*tpc_rad, 1.2*tpc_rad])
    ax.set_ylim([-1.2*tpc_rad, 1.2*tpc_rad])
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ [mm]')
    ax.set_ylabel('$y$ [mm]')
    ax.add_artist(Circle((0, 0), tpc_rad*1.02, fc='none', ec='k', lw=2))

    for k, pmt_i in enumerate(pmt_indices[event_id]):
        pos = pmt_pos_array[pmt_i]
        ax.add_artist(Circle(pos, pmt_diam/2., fc=sm.to_rgba(counts[event_id, pmt_i])))

    return fig, ax

def rotate_points(points, center, angle_deg):
    pts = points - center
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s],[s, c]])
    return pts @ R.T + center

def hex_oriented_indices_in_rect(xy, diam, width, height, center=(0.0, 0.0),
                                 angles=(0, 60, 120, 180, 240, 300)):
    """For each angle, rotate the hex grid about `center`, select PMTs inside an
    axis-aligned rectangle of size (width x height) in the rotated frame,
    and return indices sorted row-major in that frame.

    :param xy: PMT coordinates in original build order
    :type xy: (N,2) ndarray
    :param diam: PMT pitch along in-row axis (your generator's `diam`)
    :type diam: float
    :param width: Rectangle width (x-extent) in the rotated frame
    :type width: float
    :param height: Rectangle height (y-extent) in the rotated frame
    :type height: float
    :param center: Rotation and rectangle center (default (0,0))
    :type center: (2,) tuple
    :param angles: Rotation angles in degrees (multiples of 60 recommended)
    :type angles: iterable of ints
    :return: For each angle, indices of PMTs inside the rotated rectangle, sorted row-major in that rotated frame
    :rtype: dict[int, np.ndarray]
    :return: For each angle, a boolean mask over PMTs indicating inclusion
    :rtype: dict[int, np.ndarray (bool)]
    """
    xy = np.asarray(xy, dtype=float)
    center = np.asarray(center, dtype=float)

    pitch_x = diam*np.sqrt(3)/2.0  # row-to-row spacing
    pitch_y = diam                 # in-row spacing
    hx, hy = 0.5*width, 0.5*height
    eps = 1e-9

    orders, masks = {}, {}
    for angle in angles:
        xy_rot = rotate_points(xy, center, angle)
        xr, yr = (xy_rot - center).T

        # Snap ALL PMTs to hex lattice indices in the rotated frame
        i_row_all = np.rint(xr / pitch_x).astype(int)
        j_col_all = np.rint((yr - (i_row_all & 1)*0.5*diam) / pitch_y).astype(int)

        # Convert metric width/height to integer half-extents in row/col units
        max_i = int(np.floor(hx / pitch_x - eps))
        max_j = int(np.floor(hy / pitch_y - eps))

        # Membership decided in index space → no off-by-one across angles
        in_rect = (np.abs(i_row_all) <= max_i) & (np.abs(j_col_all) <= max_j)
        if not np.any(in_rect):
            orders[angle] = np.array([], dtype=int)
            masks[angle] = in_rect
            continue

        # Use the snapped indices for the selected subset
        i_row = i_row_all[in_rect]
        j_col = j_col_all[in_rect]
        xr_sel, yr_sel = xr[in_rect], yr[in_rect]
        idx_sel = np.nonzero(in_rect)[0]

        # Sort row-major in rotated frame: primary row, then col; tie-break by (yr,xr)
        sort_keys = (
            xr_sel,                  # least important
            yr_sel,
            j_col.astype(np.int64),
            i_row.astype(np.int64)   # most important
        )
        order_local = np.lexsort(sort_keys)
        orders[angle] = idx_sel[order_local]
        masks[angle] = in_rect

    return orders, masks

def hex_rowcol_indices(xy, diam, center=(0.0, 0.0), angle_deg=0.0):
    """Compute integer (row, col) indices for a hex-staggered grid in a given orientation.

    :param xy: PMT coordinates in the original build frame
    :type xy: (N,2) float ndarray
    :param diam: In-row pitch used by your generator (same as PMT diameter there)
    :type diam: float
    :param center: Hex grid center (use the same center you used to generate positions)
    :type center: (2,) float tuple
    :param angle_deg: Rotation angle (deg). Use multiples of 60 for hex orientations
    :type angle_deg: float
    :return: Integer row index for each PMT in the rotated frame
    :rtype: (N,) int ndarray
    :return: Integer column index for each PMT in the rotated frame (row-dependent offset handled)
    :rtype: (N,) int ndarray
    """
    xy = np.asarray(xy, dtype=float)
    center = np.asarray(center, dtype=float)
    # rotate into the target orientation
    xy_rot = rotate_points(xy, center, angle_deg)
    xr, yr = (xy_rot - center).T

    pitch_x = diam * np.sqrt(3) / 2.0   # row-to-row spacing (your i-step)
    pitch_y = diam                      # in-row spacing (your j-step)
    eps = 1e-9

    # snap x to integer rows; odd rows are y-shifted by half a diam
    i_row = np.rint(xr / pitch_x + eps).astype(int)
    j_col = np.rint((yr - (i_row & 1) * 0.5 * diam) / pitch_y + eps).astype(int)
    return i_row, j_col

def group_indices_by_row_in_rect(xy, diam, width, height,
                                 center=(0.0, 0.0), angle_deg=0.0):
    """Rotate coordinates by angle_deg about `center`, select only PMTs inside an
    axis-aligned rectangle (width x height) in that rotated frame, and group
    their ORIGINAL indices by hex row, ordered by column (row-major).

    :param xy: PMT coordinates in original build order
    :type xy: (N,2) float ndarray
    :param diam: In-row pitch (your generator's `diam`)
    :type diam: float
    :param width: Rectangle width in the rotated frame
    :type width: float
    :param height: Rectangle height in the rotated frame
    :type height: float
    :param center: Rotation and rectangle center
    :type center: (2,) tuple
    :param angle_deg: Rotation angle in degrees (use multiples of 60 for hex orientations)
    :type angle_deg: float
    :return: Mapping: row_id -> array of ORIGINAL PMT indices in that row, sorted left->right (by j_col) within the rotated rectangle. Rows that have no PMTs in the rectangle are omitted.
    :rtype: dict[int, np.ndarray]
    :return: Boolean mask over PMTs indicating inclusion in the rectangle
    :rtype: (N,) bool ndarray
    """
    xy = np.asarray(xy, dtype=float)
    center = np.asarray(center, dtype=float)

    # Rotate and translate to center
    xy_rot = rotate_points(xy, center, angle_deg)
    xr, yr = (xy_rot - center).T

    # Rectangle membership in rotated frame
    hx, hy = 0.5*width, 0.5*height

    # Hex snapping in rotated frame
    pitch_x = diam*np.sqrt(3)/2.0   # row-to-row spacing
    pitch_y = diam                  # in-row spacing
    eps = 1e-9

    # Snap ALL PMTs to hex lattice indices in the rotated frame
    i_row_all = np.rint(xr / pitch_x).astype(int)
    j_col_all = np.rint((yr - (i_row_all & 1)*0.5*diam) / pitch_y).astype(int)

    # Convert metric width/height to integer half-extents in row/col units
    max_i = int(np.floor(hx / pitch_x - eps))
    max_j = int(np.floor(hy / pitch_y - eps))

    # Membership decided in index space → no off-by-one across angles
    in_rect = (np.abs(i_row_all) <= max_i) & (np.abs(j_col_all) <= max_j)

    # Use the snapped indices for the selected subset
    i_row = i_row_all[in_rect]
    j_col = j_col_all[in_rect]
    xr_sel, yr_sel = xr[in_rect], yr[in_rect]
    idx_sel = np.nonzero(in_rect)[0]

    i_row = np.rint(xr_sel / pitch_x + eps).astype(int)
    j_col = np.rint((yr_sel - (i_row & 1)*0.5*diam) / pitch_y + eps).astype(int)

    # Stable row-major ordering: primary by row, then by col; tie-break by (yr,xr)
    order_local = np.lexsort((xr_sel, yr_sel, j_col.astype(np.int64), i_row.astype(np.int64)))
    i_sorted = i_row[order_local]
    idx_sorted = idx_sel[order_local]
    j_sorted = j_col[order_local]

    # Group consecutive equal rows
    rows = {}
    start = 0
    n = i_sorted.size
    while start < n:
        r = i_sorted[start]
        end = start + 1
        while end < n and i_sorted[end] == r:
            end += 1
        # Within the row, idx_sorted is already ordered by j_col (then yr,xr)
        rows[r] = idx_sorted[start:end]
        start = end

    return rows

def images_from_events(counts, pmt_pos_array, pmt_diam, width, height, num_images=10000000):
    angles = np.arange(0, 360, 60)

    images = []
    pmt_indices = []

    for event_id in range(len(counts)):

        if np.sum(counts[event_id]) == 0:
            continue

        center = np.sum(pmt_pos_array*counts[event_id, :, None], axis=0)/np.sum(counts[event_id])

        orders, masks = hex_oriented_indices_in_rect(pmt_pos_array, diam=pmt_diam, width=width, \
                                                     height=height, center=center)
        
        counts_images = []
        angle_indices = []
        angle_row_indices = []

        for angle in angles:
            idx = orders[angle]

            i_row, j_col = hex_rowcol_indices(xy=pmt_pos_array, diam=pmt_diam, center=center, angle_deg=angle)
            rows = group_indices_by_row_in_rect(pmt_pos_array, diam=pmt_diam, width=width, \
                                                height=height, center=center, angle_deg=angle)

            image_rows = len(rows)
            image_cols = min([len(rows[k]) for k in rows.keys()])
            counts_image = np.zeros((image_rows, image_cols))
            for r, k in enumerate(rows.keys()):
                for c in range(counts_image.shape[1]):
                    if len(rows[k]) > image_cols:
                        counts_image[r, c] = np.mean(counts[event_id][rows[k]][c:c+1])
                    else:
                        counts_image[r, c] = counts[event_id][rows[k]][c]

            counts_images.append(counts_image/np.sum(counts_image))
            angle_indices.append(idx)
            angle_row_indices.append(i_row)

        try:
            counts_images = np.array(counts_images)
            angle_indices = np.array(angle_indices)
            angle_row_indices = np.array(angle_row_indices)
        except ValueError:
            continue

        x_image_vals = np.arange(counts_images.shape[1], dtype=float) - (counts_images.shape[1] - 1)/2.
        # counts_images_shifted = counts_images #- np.mean(counts_images, axis=(1,2), keepdims=True)
        second_moments = np.sum(counts_images*x_image_vals[None, :, None]**2, axis=(1,2))
        best_angle_ind = np.argmax(second_moments)

        images.append(counts_images[best_angle_ind])
        pmt_indices.append(angle_indices[best_angle_ind])

        shapes = [np.shape(im) for im in images]
        np.array(shapes)[:, 0]
        np.array([])

        if len(images) >= num_images:
            break

    return np.array(images), np.array(pmt_indices)

def _process_single_event(event_data):
    """Process a single event for parallel execution.
    
    :param event_data: Tuple containing (event_id, counts_single_event, pmt_pos_array, pmt_diam, width, height)
    :type event_data: tuple
    :return: Tuple containing (event_id, image, pmt_indices) or None if event should be skipped
    :rtype: tuple or None
    """
    event_id, counts_single_event, pmt_pos_array, pmt_diam, width, height = event_data
    
    if np.sum(counts_single_event) == 0:
        return None

    center = np.sum(pmt_pos_array*counts_single_event[:, None], axis=0)/np.sum(counts_single_event)
    angles = np.arange(0, 360, 60)

    orders, masks = hex_oriented_indices_in_rect(pmt_pos_array, diam=pmt_diam, width=width, \
                                                 height=height, center=center)
    
    counts_images = []
    angle_indices = []
    angle_row_indices = []

    for angle in angles:
        idx = orders[angle]

        i_row, j_col = hex_rowcol_indices(xy=pmt_pos_array, diam=pmt_diam, center=center, angle_deg=angle)
        rows = group_indices_by_row_in_rect(pmt_pos_array, diam=pmt_diam, width=width, \
                                            height=height, center=center, angle_deg=angle)

        image_rows = len(rows)
        image_cols = min([len(rows[k]) for k in rows.keys()])
        counts_image = np.zeros((image_rows, image_cols))
        for r, k in enumerate(rows.keys()):
            for c in range(counts_image.shape[1]):
                if len(rows[k]) > image_cols:
                    counts_image[r, c] = np.mean(counts_single_event[rows[k]][c:c+1])
                else:
                    counts_image[r, c] = counts_single_event[rows[k]][c]

        counts_images.append(counts_image/np.sum(counts_image))
        angle_indices.append(idx)
        angle_row_indices.append(i_row)

    try:
        counts_images = np.array(counts_images)
        angle_indices = np.array(angle_indices)
        angle_row_indices = np.array(angle_row_indices)
    except ValueError:
        return None

    x_image_vals = np.arange(counts_images.shape[1], dtype=float) - (counts_images.shape[1] - 1)/2.
    # counts_images_shifted = counts_images - np.mean(counts_images, axis=(1,2), keepdims=True)
    second_moments = np.sum(counts_images*x_image_vals[None, :, None]**2, axis=(1,2))
    best_angle_ind = np.argmax(second_moments)

    return (event_id, counts_images[best_angle_ind], angle_indices[best_angle_ind])

def images_from_events_parallel(counts, pmt_pos_array, pmt_diam, width, height, num_images=10000000, \
                                n_jobs=None, extra_fac=1.5):
    """Parallelized version of images_from_events using joblib.
    
    :param counts: Event count data
    :type counts: np.ndarray
    :param pmt_pos_array: PMT position array
    :type pmt_pos_array: np.ndarray
    :param pmt_diam: PMT diameter
    :type pmt_diam: float
    :param width: Image width
    :type width: float
    :param height: Image height
    :type height: float
    :param num_images: Maximum number of images to process, defaults to 10000000
    :type num_images: int, optional
    :param n_jobs: Number of jobs to run in parallel, defaults to -1 (all CPUs)
    :type n_jobs: int, optional
    :return: Tuple containing (images, pmt_indices)
    :rtype: tuple
    """
    if n_jobs is None:
        n_jobs = -1  # Use all available CPUs
    
    # Filter out events with zero counts first
    valid_events = []
    for i in range(len(counts)):
        if np.sum(counts[i]) > 0:
            valid_events.append(i)
    
    # Limit to num_images if we have enough valid events
    if len(valid_events) > num_images:
        valid_events = valid_events[:int(num_images*extra_fac)]
    
    # Prepare data for parallel processing - only send what each worker needs
    event_data = [(i, counts[i], pmt_pos_array, pmt_diam, width, height) for i in valid_events]
    
    # Process all events in parallel at once (no batching)
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(_process_single_event)(data) for data in event_data
    )
    
    # Collect valid results
    images = []
    pmt_indices = []
    event_ids = []
    
    for result in results:
        if result is not None:
            event_id, image, pmt_idx = result
            images.append(image)
            pmt_indices.append(pmt_idx)
            event_ids.append(event_id)
    
    return np.array(images)[:num_images], np.array(pmt_indices)[:num_images], np.array(event_ids)[:num_images]
