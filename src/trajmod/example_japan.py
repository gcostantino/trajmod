import datetime

import numpy as np

from trajmod.events import CatalogFetcher
from trajmod.events.amplitude_threshold_selector import AmplitudeThresholdSelector
from trajmod.model.model import TrajectoryModel, ModelConfig
from trajmod.model.nonlinear_prior import save_prior, load_prior
from trajmod.visualization import TrajectoryVisualizer


def ymd_decimal_year_lookup(from_decimal=False):
    import re
    """Returns a lookup table for (year, month, day) to decimal year, with the convention introduced by Nasa JPL."""
    ymd_decimal_lookup = dict()
    with open('/Users/giuseppe/Documents/DATA/common/date_utils/decyr.txt', 'r') as f:
        next(f)
        for line in f:
            line = re.sub(' +', ' ', line)
            splitted_line = line.split(' ')
            decimal, year, month, day = splitted_line[1], splitted_line[2], splitted_line[3], splitted_line[4]
            decimal, year, month, day = float(decimal), int(year), int(month), int(day)
            ymd_decimal_lookup[(year, month, day)] = decimal
    if not from_decimal:
        return ymd_decimal_lookup
    else:
        inv_lookup = {v: k for k, v in ymd_decimal_lookup.items()}
        return inv_lookup


if __name__ == '__main__':

    component = 0
    # station_code = 'J214'
    # station_coordinates = np.array([36.800, 140.754])  # lat lon

    # station_code = 'J156'
    # station_coordinates = np.array([40.515, 141.511])  # lat lon

    station_code = 'J550'
    station_coordinates = np.array([38.301, 141.501])  # lat lon

    min_time, max_time = 2000, 2025

    amplitude_selector = False

    # Configure
    config = ModelConfig(
        d_param=.8,
        include_seasonal=True,
        postseismic_mag_threshold=6.5,
        postseismic_models=['log', 'exp'],
        postseismic_allow_combined=True,
        postseismic_pair_min_mag=8.5,
        acceleration_term=True,

        tau_grid=[7, 14, 30, 60, 90, 180],
        tau_grid_exp=[180, 360, 720, 1800],

        sse_optimize_timing=True,
        # sse_timing_tol_days=20.,  # larger tolerance -> larger shrinking factor
        sse_timing_prior=False,
        sse_timing_tol_frac=1.,  # tolerance = 15 d for a 30-d event
        sse_timing_tol_max_days=120,  # cap, default is fine
        sse_min_duration_days=3,  # stops it collapsing to a step
        sse_timing_max_rounds=3,
        sse_timing_allow_overlap=False,
    )

    date_lookup = ymd_decimal_year_lookup(from_decimal=True)

    data = np.loadtxt(f'/Users/giuseppe/Documents/DATA/GNSS/GNSS_JAPAN/txt_with_unc/{station_code}.txt')

    time, data_enu, unc_enu = data[:, 0], data[:, 1:4], data[:, 4:]
    # data_enu, unc_enu = data_enu[:, :2], unc_enu[:, :2]  # keep only horizontal
    res_vel, res_vel_unc = [], []

    ts, unc = data_enu[:, component], unc_enu[:, component]

    start_date, end_date = date_lookup[time.min()], date_lookup[time.max()]
    t_dtime = np.array([datetime.datetime(*date_lookup[dcyr]) for dcyr in time])

    '''plt.plot(time, ts)
    plt.show()'''

    # load steps from master step file (exclude EQ)
    steps = np.loadtxt('/Users/giuseppe/Documents/DATA/GNSS/NGL_info/steps.txt', dtype=str, usecols=(0, 1, 2, 3))
    sta_steps = steps[steps[:, 0] == station_code]
    sta_steps = sta_steps[sta_steps[:, 1] == '1']  # Elevation_Cutoff_Changed

    step_catalog = [
        {'date': datetime(*date_lookup[dcyr]), 'event_type': 'antenna_change'} for dcyr in sta_steps[:, 1] if
        (dcyr > min_time and dcyr < max_time)]

    station_lat, station_lon = float(station_coordinates[0]), float(station_coordinates[1])

    time_filter = np.logical_and(time >= min_time, time < max_time)
    time = time[time_filter]
    t_dtime = t_dtime[time_filter]
    ts = ts[time_filter]
    unc = unc[time_filter]

    ts, unc = ts * 1000, unc * 1000  # mm

    fetcher = CatalogFetcher()
    '''eq_catalogue = fetcher.fetch_usgs_box(
        minlat=30.0, maxlat=45.0,
        minlon=130.0, maxlon=145.0,
        start_date="2020-01-01", end_date="2023-12-31",
        min_magnitude=5.0
    )'''

    eq_catalogue = fetcher.fetch_usgs(  # radius
        lat=station_lat, lon=station_lon,
        radius_km=1000.0,
        start_date="2000-01-01", end_date="2025-12-31",
        min_magnitude=6.0
    )
    # manual selection (temporary) of SSEs along the Japan Trench from visual inspection
    # I use station latitude and longitude as source location as a temporary option
    sse_tmp_dates = [(2002.9514, 2003.0034), (2012.0164, 2012.1013), (2021.1198, 2021.1882)]
    sse_cat = []
    for i in range(len(sse_tmp_dates)):
        sse_start, sse_end = sse_tmp_dates[i]  # lat/lon of centroid
        sse_cat.append({'start': datetime.datetime(*date_lookup[sse_start]),
                        'end': datetime.datetime(*date_lookup[sse_end]),
                        'lat': station_lat, 'lon': station_lon, 'magnitude': 6.})  # tmp magnitude
    print(sse_cat)

    # Create model
    model = TrajectoryModel(
        t=t_dtime, y=ts, sigma_y=unc,
        station_lat=station_lat, station_lon=station_lon,
        sse_catalog=sse_cat,
        eq_catalog=eq_catalogue,
        step_catalog=step_catalog,
        config=config,
        nonlinear_prior=None if component == 0 else load_prior(f'{station_code}_prior.json')
    )
    print(model.G)
    # selector = KneeEventSelector()
    # selected = model.select_events(selector, event_type='eq')

    '''selector = KneeEventSelector(
        criterion='bic',
        smooth_curve=True,  # Smooth noisy curves
        smooth_window=5,  # Window size
        min_amplitude=2.0  # Filter events < 2 mm
    )

    selected = model.select_events(selector, event_type='eq', plot=True)
    selector.plot_selection(save_path='event_sel.png')'''

    '''selector = AmplitudeThresholdSelector(
        threshold=3.0,  # SNR > 3 (3-sigma)
        mode='snr'
    )

    selected = model.select_events(selector, event_type='eq')
    selector.plot_amplitudes(save_path='event_snr.png')'''

    # knee = KneeEventSelector(criterion='bic', smooth_curve=True, smooth_window=7)
    # snr = AmplitudeThresholdSelector(threshold=3.0, mode='snr')

    # selector = CompositeEventSelector([knee], mode='intersection')
    if amplitude_selector:
        selector = AmplitudeThresholdSelector(
            threshold=2,  # mm
            mode='absolute'
        )

        selected = model.select_events(selector, event_type='eq')
        selector.plot_amplitudes(save_path='event_snr.png')

        selected_indices = model.select_events(
            selector=selector,
            event_type='eq',  # or 'eq' for earthquakes
            plot=True,  # Show diagnostic plots
            save_plot='selection.png'
        )
        print(selected_indices)

    # original_eq = model.selected_eq.copy()  # Keep backup
    # model.selected_eq = [original_eq[i] for i in selected_indices]

    # fig, axes = selector.plot_amplitudes()
    # plt.show()

    # model.G = None  # Force rebuild
    print(model.G)
    results = model.fit(compute_uncertainty=True)

    if component == 0:
        prior = model.export_nonlinear()
        save_prior(prior, f'{station_code}_prior.json')

    # Visualize
    viz = TrajectoryVisualizer()
    viz.plot_fit(model, figsize=(12, 6), show_uncertainty=True, show_residuals=True,
                 show_acf=False, show_histogram=False, show_qq=False,
                 save_path=f'fit_{station_code}.pdf', errorbar_nsigma=1.)
    viz.plot_fit_windowed(model, window_size_years=1., overlap_years=0., save_dir='./windows/')

    viz.plot_coefficients(results, top_n=20, save_path='coeffs.png')
    viz.plot_components(model, save_path='components.png')
    # viz.plot_correlation_matrix(results, save_path='correlation_matrix.png')
    # viz.plot_acf(results, save_path='acf.png')
    viz.plot_periodogram(results, save_path='periodogram.png')
    viz.plot_residuals_vs_fitted(results, save_path='residuals.png')

    # Check uncertainty
    if results.metadata['uncertainty']:
        unc = results.metadata['uncertainty']
        print("Standard Errors:", unc.standard_errors)
        print("VIF (multicollinearity):", unc.variance_inflation_factors)
