from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
import time
import json

# Replace this with the path to your ChromeDriver
CHROMEDRIVER_PATH = "C:/chromedriver-win64/chromedriver.exe"

def scrape_github_repositories(base_url):
    driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH))
    driver.get(f"{base_url}?tab=repositories")
    time.sleep(2)  # Allow page to load

    repositories_data = []

    while True:
        # Find all repository links on the current page
        repo_links = driver.find_elements(By.XPATH, '//*[@id="user-repositories-list"]/ul/li/div[1]/div[1]/h3/a')
        
        for repo_link in repo_links:
            repo_url = repo_link.get_attribute('href')
            driver.execute_script("window.open('{}');".format(repo_url))
            driver.switch_to.window(driver.window_handles[1])
            time.sleep(2)  # Allow repo page to load

            # Extract repository details
            repo_data = {}
            
            try:
                # Full Title Content
                title_element = driver.find_element(By.XPATH, '//*[@id="repository-container-header"]/div[1]/div[1]/div/strong/a')
                repo_data['title'] = title_element.text.strip()
            except NoSuchElementException:
                repo_data['title'] = None

            try:
                # Full README Content
                readme_element = driver.find_element(By.XPATH, '//*[@id="repo-content-pjax-container"]/div/div/div/div[1]/react-partial/div/div/div[3]/div[2]/div/div[2]')
                repo_data['readme'] = readme_element.text.strip()  # Store the full README content
            except NoSuchElementException:
                repo_data['readme'] = None

            try:
                # Languages
                languages_elements = driver.find_elements(By.XPATH, '//ul[@class="list-style-none"]/li')
                repo_data['languages'] = [lang.text.strip() for lang in languages_elements]
            except NoSuchElementException:
                repo_data['languages'] = None

            repositories_data.append(repo_data)

            # Close repository tab and return to repositories list
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
        
        # Check for "Next" button and navigate to the next page
        try:
            next_button = driver.find_element(By.XPATH, '//a[contains(@rel, "next")]')
            next_button.click()
            time.sleep(2)  # Allow next page to load
        except NoSuchElementException:
            break  # No more pages, exit the loop

    driver.quit()
    return repositories_data


if __name__ == "__main__":
    # List of base URLs of GitHub profiles (change/add URLs as needed)
    github_base_urls = [
        "https://github.com/manisha03gupta",
    ]
    
    print("Scraping data...")
    all_data = {}

    for base_url in github_base_urls:
        print(f"Scraping repositories for: {base_url}")
        all_data[base_url] = scrape_github_repositories(base_url)

    # Save data to a JSON file
    with open("github_repositories_full.json", "w") as file:
        json.dump(all_data, file, indent=4)

    print("Scraping complete. Data saved to 'github_repositories_full.json'.")
