import os
import asyncio
import xml.etree.ElementTree as ET
import aiohttp
from aiolimiter import AsyncLimiter
import pandas as pd


async def download_xml(year, semaphore):
    """
    Asynchronous function to fetch single-year expense data
    ARGS:
        year (int): year of expenses
        semaphore (asyncio.locks.Semaphore object): internal counter
        which is decremented by each acquire() call and incremented by
        each release() call; is defined in fetch_expenses() function
    RETURNS:
        xml (file): content of single-year URL named as
        'despesas_gabinetes_*.xml'
    """
    limiter = AsyncLimiter(1, 0.125)
    USER_AGENT = ''
    headers={'User-Agent': USER_AGENT}
    DATA_DIR = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    url = f'https://www.al.sp.gov.br/repositorioDados/deputados/despesas_gabinetes_{str(year)}.xml'
    async with aiohttp.ClientSession(headers=headers) as session:
        await semaphore.acquire()
        async with limiter:
            async with session.get(url) as resp:
                content = await resp.read()
                semaphore.release()
                file = f'despesas_gabinetes_{str(year)}.xml'
                with open(os.path.join(DATA_DIR, file), 'wb') as f:
                    f.write(content)


async def fetch_expenses(year_start, year_end):
    """
    Iterates over years to call 'async_fect()' function
    ARGS:
        year_start (int): first year of the timeframe
        last_year (int): last year of the timeframe
    """
    tasks = set()
    semaphore = asyncio.Semaphore(value=10)
    for i in range(int(year_start), int(year_end) + 1):
        task = asyncio.create_task(download_xml(i, semaphore))
        tasks.add(task)
    await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)


def parse_data(list_files):
    """
    Parses data from 'despesa' element within xml files into a list of
    dicts
    ARGS:
        list_files (list of strings): collection of xml files
    RETURNS:
        data (list of dicts): collection of dicts
    """
    data = list()
    for file in list_files:
        tree = ET.parse(file)
        xroot = tree.getroot()
        for child in xroot.iter('despesa'):
            cols = [elem.tag for elem in child]
            values = [elem.text for elem in child]
            data.append(dict(zip(cols, values)))
    return data


# TESTING
if __name__ == '__main__':
    import glob
    asyncio.run(fetch_expenses(2013, 2022))
    if os.path.exists(os.path.join(os.getcwd(), 'data')):
        os.chdir('data')
        files = glob.glob('*.xml')
        load = parse_data(files)
        dictio = pd.DataFrame.from_dict(load) # type: ignore
        dictio.to_csv('2013_2022.csv', index=False, encoding='utf-8', sep=',')