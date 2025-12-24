from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from time import sleep

driver = webdriver.Chrome()

try:
    # Navigate to the USGS Earth Explorer URL
    driver.get("https://ers.cr.usgs.gov/login?RET_ADDR=https%3A%2F%2Fearthexplorer.usgs.gov%2F")

    # Wait for the username field to become available (login modal appears)
    username = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.NAME, "username"))
    )
    print(username)
    username.send_keys("firdavsn")  # Replace 'your_username' with the actual username

    # Find the password field and enter the password
    password = driver.find_element(By.NAME, "password")
    password.send_keys("Fird@vs0406n")  # Replace 'your_password' with the actual password

    # Find the submit button and click it to log in
    submit_button = driver.find_element(By.ID, "loginButton")
    submit_button.click()

    # sleep(3)
    
    datasets_tab = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "tab2"))
    )
    datasets_tab.click()

    # sleep(1)
    
    landsat_button = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div[3]/div[1]/ul/li[14]/span/div/strong"))
    )
    landsat_button.click()
    # sleep(0.5)
    
    level1_button = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "cat_2337"))
    )
    level1_button.click()
    # sleep(0.5)
    
    landsat89_button = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "coll_5e81f14f59432a27"))
    )
    landsat89_button.click()
    sleep(0.5)
    
    sleep(10)
    
finally:
    # Close the browser window
    driver.quit()