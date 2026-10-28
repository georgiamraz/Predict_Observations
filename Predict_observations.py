
#imports needed
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
import astropy.units as u
import matplotlib.dates as mdates

def test_function (x):
    x = x + 1
    return x


def predict_transits(planet_name, start_date, end_date, telescope_location, baseline_duration=None, moon_restriction=0, plot=False, T0=None, P=None, D=None, airmass_limit=2):

    """
    This module contains functions to predict the transits of exoplanets based on their orbital parameters
    and observational constraints. It uses data from the NASA Exoplanet Archive and calculates visibility
    based on the observer's location, time, and other factors.

    Functions:
        predict_transits: Predicts the visible transits of an exoplanet within a given date range.

    Function Details:
        predict_transits(planet_name, start_date, end_date, telescope_location, baseline_duration=None, plot=False, T0=None, P=None, D=None):
            Predicts the visible transits of an exoplanet based on its orbital parameters and observational constraints.

            Parameters:
                planet_name (str): The name of the exoplanet to predict transits for.
                start_date (str): The start date of the observation window (format: 'YYYY-MM-DD').
                end_date (str): The end date of the observation window (format: 'YYYY-MM-DD').
                telescope_location (str): The name of the telescope location (e.g., 'Canada-France-Hawaii Telescope').
                baseline_duration (float, optional): Additional time (in days) to include before and after the transit duration.
                plot (bool, optional): Whether to generate plots of the predicted transits. Default is False.
                T0 (float, optional): The mid-transit time (BJD). If not provided, it is fetched from the NASA Exoplanet Archive.
                P (float, optional): The orbital period of the exoplanet (in days). If not provided, it is fetched from the NASA Exoplanet Archive.
                D (float, optional): The transit duration (in days). If not provided, it is fetched from the NASA Exoplanet Archive.
                moon_restriction (float): Minimum allowable moon illumination (fractional, 0-1) for the observation.
                airmass_limit(float): The maximum airmass allowed for the observation. Default is 2.

            Returns:
                tuple:
                    - start_times_utc (list of datetime): List of start times (UTC) for visible transits.
                    - end_times_utc (list of datetime): List of end times (UTC) for visible transits.

            Notes:
                - The function calculates the visibility of transits based on the observer's location, the altitude of the planet,
                and the position of the Sun.
                - Transits are considered visible if the planet is above the horizon, the Sun is below -12 degrees altitude
                (nautical twilight), and the airmass is within acceptable limits.
                - If `plot` is set to True, the function generates plots showing the airmass variation during each transit.
    """

    # Query NASA Exoplanet Archive for the planet's data
    table = NasaExoplanetArchive.query_object(planet_name, table="pscomppars")
    if T0 is None:
        T0 = table["pl_tranmid"][0].value  # Mid-transit time (BJD)
        print('T0 is',T0)
    if P is None:
        P = table["pl_orbper"][0].value  # Orbital period
        print('P is',P)
    if D is None:
        D = table["pl_trandur"][0].value / 24.0
        print('Duration is',D*24)

    # Define the planet's sky coordinates
    # planet = SkyCoord.from_name(planet_name)
    ra = table["ra"][0].value    # Right ascension (degrees)
    dec = table["dec"][0].value  # Declination (degrees)
    
    ra = table["ra"][0].value    # Right ascension (degrees)
    dec = table["dec"][0].value  # Declination (degrees)

    planet = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    
    ra_sexagesimal = planet.ra.to_string(unit=u.hourangle, sep=":", precision=2, pad=True)
    dec_sexagesimal = planet.dec.to_string(unit=u.deg, sep=":", precision=2, pad=True, alwayssign=True)

    print(f"Coordinates are RA: {ra_sexagesimal}, Dec: {dec_sexagesimal}")    # print('coordinate are',ra,dec)

    # planet = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # Convert start and end dates to JD
    start_jd = Time(start_date).jd
    end_jd = Time(end_date).jd

    # Compute transit epochs within the given time range
    E_start = int((start_jd - T0) / P)
    E_end = int((end_jd - T0) / P)
    future_transits = [T0 + (E * P) for E in range(E_start, E_end + 1)]

    # Observer location
    observer = EarthLocation.of_site(telescope_location)

    visible_transits = []
    start_times_utc = []  # Initialize list for start times
    end_times_utc = []    # Initialize list for end times

    for transit_mid in future_transits:
        if baseline_duration:
            total_duration = D + (2 * baseline_duration)
        else:
            total_duration = D

        transit_start = transit_mid - (total_duration / 2)
        transit_end = transit_mid + (total_duration / 2)

        time_start = Time(transit_start, format="jd")
        time_mid = Time(transit_mid, format="jd")
        time_end = Time(transit_end, format="jd")

        altaz_start = AltAz(obstime=time_start, location=observer)
        altaz_mid = AltAz(obstime=time_mid, location=observer)
        altaz_end = AltAz(obstime=time_end, location=observer)

        planet_altaz_start = planet.transform_to(altaz_start)
        planet_altaz_mid = planet.transform_to(altaz_mid)
        planet_altaz_end = planet.transform_to(altaz_end)

        sun_alt_start = get_sun(time_start).transform_to(altaz_start).alt
        sun_alt_end = get_sun(time_end).transform_to(altaz_end).alt
        
        # Calculate moon illumination
        moon = get_moon(time_mid, location=observer)
        sun = get_sun(time_mid)
        phase_angle = moon.separation(sun)  # Angle between moon and sun as seen from Earth
        # print('phase_angle',phase_angle.radian)
        moon_illumination = 0.5 * (1 - np.cos(phase_angle.radian))  # Fractional illumination (0-1)
        # print('moon illumination',moon_illumination)

        start_airmass = planet_altaz_start.secz
        mid_airmass = planet_altaz_mid.secz
        end_airmass = planet_altaz_end.secz

        if mid_airmass > 0 and planet_altaz_start.alt > 0 * u.deg and planet_altaz_mid.alt > 0 * u.deg \
                and planet_altaz_end.alt > 0 * u.deg and sun_alt_start < -10 * u.deg and sun_alt_end < -10 * u.deg and moon_illumination >= moon_restriction:
            if mid_airmass < airmass_limit and start_airmass < airmass_limit and end_airmass < airmass_limit:
                visible_transits.append((time_start.datetime, time_mid.datetime, time_end.datetime, mid_airmass,moon_illumination))
                start_times_utc.append(time_start.datetime)
                end_times_utc.append(time_end.datetime)

    # Print results
    if visible_transits:
        print(f"\nUpcoming transits for {planet_name} at {telescope_location}:")
        for i, (start_transit_utc, mid_transit_utc, end_transit_utc, mid_airmass,moon_illumination) in enumerate(visible_transits):
            print(f"Transit {i+1}: {start_transit_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                  f"{mid_transit_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                  f"{end_transit_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | Airmass: {mid_airmass:.3f} | Moon Illumination: {moon_illumination:.3f}")
    else:
        print("No visible transits found in the given date range.")

    # # Plotting
    # if plot and visible_transits:
    #     for i, (start_transit, mid_transit, end_transit, mid_airmass, moon_illumination) in enumerate(visible_transits):
    #         plt.figure(figsize=(10, 6))  # Create a new figure for each transit

    #         # Convert datetime to astropy Time
    #         start_time = Time(start_transit)
    #         end_time = Time(end_transit)

    #         # Compute baseline start and end if applicable
    #         if baseline_duration:
    #             baseline_start_time = start_time - baseline_duration * u.day
    #             baseline_end_time = end_time + baseline_duration * u.day
    #         else:
    #             baseline_start_time = start_time
    #             baseline_end_time = end_time

    #         # Convert to Julian Date before using np.linspace
    #         start_jd = baseline_start_time.jd  # Extend to baseline start
    #         end_jd = baseline_end_time.jd  # Extend to baseline end

    #         times = np.linspace(start_jd, end_jd, 150)  # Increase points for smoothness

    #         airmasses = []
    #         for t in times:
    #             time = Time(t, format='jd')  # Convert back to Time object
    #             altaz = AltAz(obstime=time, location=observer)
    #             planet_altaz = planet.transform_to(altaz)
    #             airmasses.append(planet_altaz.secz.value)

    #         # Convert times for plotting
    #         datetime_array = Time(times, format='jd').datetime  # Use .datetime instead of .to_datetime()

    #         plt.plot(datetime_array, airmasses,
    #                  label=f"Transit {i+1} | {mid_transit.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    #         # Plot vertical lines for transit events
    #         plt.axvline(start_transit, color='r', linestyle='--', label='Start of Transit' if i == 0 else "")
    #         plt.axvline(mid_transit, color='g', linestyle='--', label='Mid Transit' if i == 0 else "")
    #         plt.axvline(end_transit, color='b', linestyle='--', label='End of Transit' if i == 0 else "")

    #         # Plot baseline start and end if applicable
    #         if baseline_duration:
    #             baseline_start = baseline_start_time.datetime
    #             baseline_end = baseline_end_time.datetime

    #             plt.axvline(baseline_start, color='c', linestyle='--', label='Baseline Start' if i == 0 else "")
    #             plt.axvline(baseline_end, color='m', linestyle='--', label='Baseline End' if i == 0 else "")

    #         plt.xlabel('Time (UTC)')
    #         plt.ylabel('Airmass')
    #         plt.legend()
    #         plt.show()  # Show each plot separately


    
    # Plotting
    if plot and visible_transits:
        for i, (start_transit, mid_transit, end_transit, mid_airmass, moon_illumination) in enumerate(visible_transits):
            plt.figure(figsize=(10, 6))

            # Convert datetime to astropy Time
            start_time = Time(start_transit)
            end_time = Time(end_transit)

            # Compute baseline start and end
            if baseline_duration:
                baseline_start_time = start_time - baseline_duration * u.day
                baseline_end_time = end_time + baseline_duration * u.day
            else:
                baseline_start_time = start_time
                baseline_end_time = end_time

            # Generate times across the full observing window
            start_jd = baseline_start_time.jd
            end_jd = baseline_end_time.jd
            times = np.linspace(start_jd, end_jd, 300)

            # Calculate airmass
            airmasses = []
            for t in times:
                time = Time(t, format='jd')
                altaz = AltAz(obstime=time, location=observer)
                planet_altaz = planet.transform_to(altaz)
                airmasses.append(planet_altaz.secz.value)

            datetime_array = Time(times, format='jd').datetime
            airmasses = np.array(airmasses)

            # Replace below-horizon values with NaN
            airmasses[airmasses <= 0] = np.nan

            # Standard airmass plot: 1.0 at the top
            max_airmass = 1.5
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_ylim(max_airmass, 0.9)

            # Fill the entire plotting background grey
            ax.axvspan(
                baseline_start_time.datetime,
                baseline_end_time.datetime,
                color='lightgrey',
                alpha=0.8,
                label='Baseline / Out of Transit',
                zorder=0
            )

            # Fill the entire transit interval light blue
            ax.axvspan(
                start_transit,
                end_transit,
                color='lightskyblue',
                alpha=0.8,
                label='Transit',
                zorder=1
            )

            # Plot airmass curve on top of the colored background
            ax.plot(
                datetime_array,
                airmasses,
                color='black',
                linewidth=2,
                label='Airmass',
                zorder=2
            )

            # Transit event markers
            ax.axvline(start_transit, color='royalblue', linestyle='--',
                    linewidth=1.5, label='Transit Start', zorder=3)
            ax.axvline(mid_transit, color='green', linestyle='--',
                    linewidth=1.5, label='Mid-Transit', zorder=3)
            ax.axvline(end_transit, color='royalblue', linestyle='--',
                    linewidth=1.5, label='Transit End', zorder=3)

            # Baseline boundaries
            if baseline_duration:
                ax.axvline(baseline_start_time.datetime, color='dimgrey',
                        linestyle=':', linewidth=1.5, label='Obs Start')
                ax.axvline(baseline_end_time.datetime, color='dimgrey',
                        linestyle=':', linewidth=1.5, label='Obs End')

            ax.set_xlabel('Time (UTC)')
            ax.set_ylabel('Airmass')
            ax.set_title(
                f"Transit {i+1} | {mid_transit.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )

            ax.legend(loc='best')
            ax.grid(axis='y', linestyle=':', alpha=0.4)
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()


    return start_times_utc, end_times_utc



def predict_emission_observation(planet_name, start_date, end_date, telescope_location, phase, plot, additional_obs_hours, moon_restriction,T0=None, P=None, D=None, airmass_limit=2):
    """
Predict the visibility of emission observations for an exoplanet based on its orbital parameters, 
observer location, and observational constraints such as moon illumination and additional observation hours.

Parameters:
    planet_name (str): The name of the exoplanet to predict observations for.
    start_date (str): The start date of the observation window (format: 'YYYY-MM-DD').
    end_date (str): The end date of the observation window (format: 'YYYY-MM-DD').
    telescope_location (str): The name of the telescope location (e.g., 'Canada-France-Hawaii Telescope').
    phase (float): The midpoint phase of the observations (e.g., 0.25 for quadrature).
    plot (bool): Whether to generate plots of the predicted observations. Default is False.
    additional_obs_hours (float): Additional observation time (in hours) to include before and after the event.
    moon_restriction (float): Minimum allowable moon illumination (fractional, 0-1) for the observation.
    T0 (float, optional): The mid-transit time (BJD). If not provided, it is fetched from the NASA Exoplanet Archive.
    P (float, optional): The orbital period of the exoplanet (in days). If not provided, it is fetched from the NASA Exoplanet Archive.
    D (float, optional): The transit duration (in days). If not provided, it is fetched from the NASA Exoplanet Archive.
    airmass_limit(float): The maximum airmass allowed for the observation. Default is 2.

Returns:
    tuple:
        - List of start times (datetime) for visible phases.
        - List of end times (datetime) for visible phases.

Notes:
    - The function calculates the visibility of emission observations based on the observer's location, 
      the altitude of the planet, the position of the Sun, and moon illumination.
    - Observations are considered visible if the planet is above the horizon, the Sun is below -12 degrees altitude 
      (nautical twilight), and the moon illumination meets the specified restriction.
    - If `plot` is set to True, the function generates plots showing the airmass variation during each observation phase.
    """
    
    
    # Query NASA Exoplanet Archive for the planet's data
    table = NasaExoplanetArchive.query_object(planet_name, table="pscomppars")

    print('For planet',planet_name,'the following parameters are used:')
    
    if T0 is None:
        T0 = table["pl_tranmid"][0].value  # Mid-transit time (BJD)
        print('T0 is',T0)
    if P is None:
        P = table["pl_orbper"][0].value  # Orbital period
        print('P is',P)
    if D is None:
        D = table["pl_trandur"][0].value / 24.0
        print('D is',D)

    # Define the planet's sky coordinates
    # planet = SkyCoord.from_name(planet_name)
    
    ra = table["ra"][0].value    # Right ascension (degrees)
    dec = table["dec"][0].value  # Declination (degrees)

    planet = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    
    ra_sexagesimal = planet.ra.to_string(unit=u.hourangle, sep=":", precision=2, pad=True)
    dec_sexagesimal = planet.dec.to_string(unit=u.deg, sep=":", precision=2, pad=True, alwayssign=True)

    print(f"Coordinates are RA: {ra_sexagesimal}, Dec: {dec_sexagesimal}")    # print('coordinate are',ra,dec)


    # Convert start and end dates to JD
    start_jd = Time(start_date).jd
    end_jd = Time(end_date).jd
    
    # Compute transit epochs within the given time range
    E_start = int((start_jd - T0) / P)
    E_end = int((end_jd - T0) / P)
    future_phases = [T0 + (E * P) + (phase * P) for E in range(E_start, E_end + 1)]
    
    # Observer location
    observer = EarthLocation.of_site(telescope_location)

    visible_phases = []

    for phase_mid in future_phases:
        phase_start = phase_mid - (D / 2) - (additional_obs_hours / 24.0 / 2)
        phase_end = phase_mid + (D / 2) + (additional_obs_hours / 24.0 / 2)
        
        time_start = Time(phase_start, format='jd')
        time_mid = Time(phase_mid, format='jd')
        time_end = Time(phase_end, format='jd')

        altaz_start = AltAz(obstime=time_start, location=observer)
        altaz_mid = AltAz(obstime=time_mid, location=observer)
        altaz_end = AltAz(obstime=time_end, location=observer)
        
        planet_altaz_start = planet.transform_to(altaz_start)
        planet_altaz_mid = planet.transform_to(altaz_mid)
        planet_altaz_end = planet.transform_to(altaz_end)
        
        sun_alt_start = get_sun(time_start).transform_to(altaz_start).alt
        sun_alt_end = get_sun(time_end).transform_to(altaz_end).alt
    
        # Calculate moon illumination
        moon = get_moon(time_mid, location=observer)
        sun = get_sun(time_mid)
        phase_angle = moon.separation(sun)  # Angle between moon and sun as seen from Earth
        # print('phase_angle',phase_angle.radian)
        moon_illumination = 0.5 * (1 - np.cos(phase_angle.radian))  # Fractional illumination (0-1)
        # print('moon illumination',moon_illumination)
        start_airmass = planet_altaz_start.secz
        mid_airmass = planet_altaz_mid.secz
        end_airmass = planet_altaz_end.secz
        
        #debug moon iullumination
        # mid_transit_utc=time_mid.datetime
        # print(f"Moon illumination at mid-transit: {moon_illumination:.2f}", f"{mid_transit_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC |")


        if mid_airmass > 0 and planet_altaz_start.alt > 0 * u.deg and planet_altaz_mid.alt > 0 * u.deg and planet_altaz_end.alt > 0 * u.deg and sun_alt_start < -10 * u.deg and sun_alt_end < -10 * u.deg and moon_illumination >= moon_restriction:
            if mid_airmass < airmass_limit and start_airmass < airmass_limit and end_airmass < airmass_limit:
                visible_phases.append((time_start.datetime, time_mid.datetime, time_end.datetime, mid_airmass,moon_illumination))
    
    # Print results
    if visible_phases:
        print(f"\nUpcoming fully visible phase {phase} for {planet_name} at {telescope_location}:")
        for i, (start_phase_utc, mid_phase_utc, end_phase_utc, mid_airmass, moon_illumination) in enumerate(visible_phases):
            print(f"Phase {i+1}: {start_phase_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | {mid_phase_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | {end_phase_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | Mid Airmass: {mid_airmass:.3f}, | Moon Illumination: {moon_illumination:.3f}")
    else:
        print("No fully visible phases found in the given date range.")



    # Plotting
    # Plotting
    if plot and visible_phases:
        for i, (start_phase, mid_phase, end_phase, mid_airmass, moon_illumination) in enumerate(visible_phases):

            fig, ax = plt.subplots(figsize=(11, 5.5))

            # Convert times to astropy Time
            start_time = Time(start_phase)
            end_time = Time(end_phase)

            # Sample the observing interval
            times = np.linspace(start_time.jd, end_time.jd, 200)

            airmasses = []
            for t in times:
                time = Time(t, format='jd')
                altaz = AltAz(obstime=time, location=observer)
                planet_altaz = planet.transform_to(altaz)
                airmasses.append(planet_altaz.secz.value)

            datetime_array = Time(times, format='jd').datetime

            # Airmass curve
            ax.plot(
                datetime_array,
                airmasses,
                color='midnightblue',
                linewidth=2.5,
                label='Target airmass',
                zorder=3
            )

            # Light shading for the observing interval
            ax.axvspan(
                start_phase, end_phase,
                color='steelblue',
                alpha=0.08,
                zorder=1
            )

            # Airmass limit
            ax.axhline(
                2.5,
                color='gray',
                linestyle=':',
                linewidth=1.4,
                alpha=0.8,
                label='Airmass limit (2.5)'
            )

            # Mark the midpoint of the observation
            ax.axvline(
                mid_phase,
                color='darkorange',
                linestyle='--',
                linewidth=1.5,
                alpha=0.9,
                label='Mid-observation'
            )

            # Classic airmass axis
            ax.set_ylim(2.5, 0.9)
            ax.set_yticks(np.arange(1.0, 2.51, 0.25))

            # Labels and title
            ax.set_xlabel('Time (UTC)', fontsize=11)
            ax.set_ylabel('Airmass', fontsize=11)
            ax.set_title(
                f"Emission Observation {i+1} | "
                f"{mid_phase.strftime('%Y-%m-%d')}",
                fontsize=13,
                fontweight='bold',
                pad=12
            )

            # Format time axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=5, maxticks=8)
            )

            # Clean grid and borders
            ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Observation details
            ax.text(
                0.02, 0.04,
                f"Moon illumination: {moon_illumination:.0%}",
                transform=ax.transAxes,
                fontsize=9,
                color='dimgray',
                verticalalignment='bottom'
            )

            ax.legend(
                loc='upper left',
                frameon=False,
                fontsize=9
            )

            fig.tight_layout()
            plt.show()



    # Return start and end times
    return [start for start, _, _, _, _ in visible_phases], [end for _, _, end, _, _ in visible_phases]