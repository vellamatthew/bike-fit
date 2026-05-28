"""Web scraping functions for bike geometry data."""

from bs4 import BeautifulSoup
import requests


def search_bike(query: str) -> list[dict]:
    """
    Search for bikes on geometrygeeks.bike.

    Args:
        query: Search query (e.g., "ritchey ascent")

    Returns:
        List of bike dictionaries with keys: brand, model, year, url
    """
    url = f"https://geometrygeeks.bike/bike-directory/search/?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the results table
    table = soup.find('table', id='bike_table')
    if not table:
        return []

    bikes = []

    # Parse each row
    tbody = table.find('tbody')
    if not tbody:
        return []

    for row in tbody.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 3:
            continue

        bike = {
            'brand': cells[0].text.strip(),
            'model': cells[1].text.strip(),
            'year': cells[2].text.strip(),
            'url': 'https://geometrygeeks.bike' + cells[1].find('a')['href']
        }
        bikes.append(bike)

    return bikes


def get_bike_geometry(url: str) -> dict:
    """
    Extract full geometry data for a specific bike.

    Args:
        url: Full URL to bike page on geometrygeeks.bike

    Returns:
        Dictionary with keys:
            - bike_name: Full name of the bike
            - sizes: List of available sizes
            - measurements: Dict mapping measurement names to size->value dicts
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the geometry table
    table = soup.find('table', class_='bike-geometry-table')
    if not table:
        return {
            'bike_name': 'Unknown',
            'sizes': [],
            'measurements': {}
        }

    # Extract size names from header (only from the first tr, avoiding noscript)
    header_row = table.find('thead').find_all('tr')[0]  # Get first row only
    headers = header_row.find_all('th')
    sizes = [th.get_text(strip=True) for th in headers[1:]]  # Skip first empty header

    # Extract measurements
    measurements = {}
    tbody = table.find('tbody')
    if tbody:
        rows = tbody.find_all('tr')

        for row in rows:
            # Skip the compare-select row
            if 'compare-select-bar' in row.get('class', []):
                continue

            cells = row.find_all('td')
            if not cells:
                continue

            # Get measurement name from first cell
            first_cell = cells[0]

            # Check if this is a measurement row (has data-param attribute)
            if first_cell.has_attr('data-param'):
                measurement_name = first_cell.get('data-param')
                values = [cell.get_text(strip=True) for cell in cells[1:]]
                measurements[measurement_name] = dict(zip(sizes, values))

    # Get bike name
    bike_name = 'Unknown'
    h1 = soup.find('h1')
    if h1:
        bike_name = h1.get_text(strip=True)

    return {
        'bike_name': bike_name,
        'sizes': sizes,
        'measurements': measurements
    }
