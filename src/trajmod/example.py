"""

# === KNEE SELECTOR (Enhanced) ===
selector = KneeEventSelector(
    criterion='bic',       # 'aic', 'bic', or 'ssr'
    smooth_curve=True,     # Smooth noisy curves
    min_amplitude=2.0      # Filter small events
)

# === AMPLITUDE THRESHOLD ===
selector = AmplitudeThresholdSelector(
    threshold=3.0,         # Threshold value
    mode='snr'             # 'absolute', 'relative', or 'snr'
)

# === LASSO ===
selector = LassoEventSelector(cv=5)

# === MARGINAL SCREENING ===
selector = MarginalScreeningSelector(alpha=0.05, use_fdr=True)

# === COMPOSITE ===
selector = CompositeEventSelector(
    [selector1, selector2],
    mode='intersection'    # 'intersection' (AND) or 'union' (OR)
)

# === USE ANY SELECTOR ===
selected = model.select_events(selector, event_type='sse', plot=True)


"""
import matplotlib.pyplot as plt

from trajmod.events import CatalogFetcher
from trajmod.events.amplitude_threshold_selector import AmplitudeThresholdSelector
from trajmod.model.model import TrajectoryModel, ModelConfig
from trajmod.events.event_selection import KneeEventSelector, LassoEventSelector, CompositeEventSelector
from trajmod.visualization import TrajectoryVisualizer
import numpy as np
import datetime


def build_japan_catalogue_from_nied(min_time=None, max_time=None):
    eq_list = []
    cat_dates = np.loadtxt(f'/Users/giuseppe/Documents/DATA/SEISMIC_CATALOGS/JAPAN/cat_nied_small.txt',
                           skiprows=1, usecols=(0,), delimiter=" ", dtype=str)
    eq_data = np.loadtxt(f'/Users/giuseppe/Documents/DATA/SEISMIC_CATALOGS/JAPAN/cat_nied_small.txt',
                           skiprows=1, usecols=(1, 2, 11), delimiter=" ")

    for i in range(len(cat_dates)):
        year, month, day = cat_dates[i].split(',')[0].split('/')
        if min_time is not None and max_time is not None:
            if int(year) < min_time or int(year) > max_time:
                continue
        dtime = datetime.datetime(int(year), int(month), int(day))
        eq_list.append({'lat': eq_data[i][0], 'lon': eq_data[i][1], 'date': dtime, 'magnitude': eq_data[i][2]})

    return eq_list

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
    # Configure
    config = ModelConfig(
        d_param=.8,
        include_seasonal=True,
        postseismic_mag_threshold=6.5
    )

    station_code = 'J214'
    station_coordinates = np.array([36.800, 140.754])  # lat lon
    min_time, max_time = 2000, 2025

    date_lookup = ymd_decimal_year_lookup(from_decimal=False)
    lookup_decimal = ymd_decimal_year_lookup(from_decimal=True)

    component = 0
    data = np.loadtxt(f'/Users/giuseppe/Documents/DATA/GNSS/GNSS_JAPAN/txt_with_unc/{station_code}.txt')

    time, data_enu, unc_enu = data[:, 0], data[:, 1:4], data[:, 4:]
    # data_enu, unc_enu = data_enu[:, :2], unc_enu[:, :2]  # keep only horizontal
    res_vel, res_vel_unc = [], []

    ts, unc = data_enu[:, component], unc_enu[:, component]

    start_date, end_date = lookup_decimal[time.min()], lookup_decimal[time.max()]
    t_dtime = np.array([datetime.datetime(*lookup_decimal[dcyr]) for dcyr in time])

    '''plt.plot(time, ts)
    plt.show()'''

    station_lat, station_lon = float(station_coordinates[0]), float(station_coordinates[1])

    time_filter = np.logical_and(time >= min_time, time < max_time)
    time = time[time_filter]
    t_dtime = t_dtime[time_filter]
    ts = ts[time_filter]
    unc = unc[time_filter]

    ts = ts * 1000  # mm

    #eq_catalogue = build_japan_catalogue_from_nied()
    #print('# events in the EQ catalogue', len(eq_catalogue))
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

    sse_events = None


    # Create model
    model = TrajectoryModel(
        t=t_dtime, y=ts, sigma_y=unc,
        station_lat=station_lat, station_lon=station_lon,
        sse_catalog=sse_events,
        eq_catalog=eq_catalogue,
        config=config
    )
    print(model.G)
    #selector = KneeEventSelector()
    #selected = model.select_events(selector, event_type='eq')

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

    #knee = KneeEventSelector(criterion='bic', smooth_curve=True, smooth_window=7)
    #snr = AmplitudeThresholdSelector(threshold=3.0, mode='snr')

    #selector = CompositeEventSelector([knee], mode='intersection')

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

    #original_eq = model.selected_eq.copy()  # Keep backup
    #model.selected_eq = [original_eq[i] for i in selected_indices]

    #fig, axes = selector.plot_amplitudes()
    #plt.show()

    # model.G = None  # Force rebuild
    print(model.G)
    results = model.fit(compute_uncertainty=True)

    # Visualize
    viz = TrajectoryVisualizer()
    viz.plot_fit(model, show_uncertainty=True, save_path='fit.png')
    viz.plot_coefficients(results, top_n=20, save_path='coeffs.png')
    viz.plot_components(model, save_path='components.png')

    # Check uncertainty
    if results.metadata['uncertainty']:
        unc = results.metadata['uncertainty']
        print("Standard Errors:", unc.standard_errors)
        print("VIF (multicollinearity):", unc.variance_inflation_factors)
