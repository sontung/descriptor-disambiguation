import argparse
import glob

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import os


def login(driver, username, password):
    driver.get(
        "https://www.visuallocalization.net/login/"
    )  # Replace with the actual login URL

    # Find the username field and input username
    username_input = driver.find_element(By.NAME, "username")  # Using 'name' attribute
    username_input.send_keys(username)  # Replace with your username

    # Find the password field and input password
    password_input = driver.find_element(By.NAME, "password")  # Using 'name' attribute
    password_input.send_keys(password)  # Replace with your password

    login_button = driver.find_element(
        By.XPATH, "//button[@type='submit']"
    )  # Locate the submit button
    login_button.click()

    # Wait for the login to complete (adjust time based on site speed)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.XPATH, "//button[@type='submit']")
        )  # Replace with an element that loads after login
    )


def extract_numbers(input_string):
    if input_string is None:
        return None
    # Use regex to find all numbers in the format of floating-point numbers
    numbers = re.findall(r"\d+\.\d+", input_string)
    return [float(num) for num in numbers]


def get_present_substrings(
    main_string,
    substrings1=["d2net", "r2d2", "superpoint", "dog"],
    substrings2=["salad", "eigenplaces", "mixvpr"],
):
    local = [substring for substring in substrings1 if substring in main_string]
    glo = [substring for substring in substrings2 if substring in main_string]
    local.extend(glo)
    return local


def view(username, password):
    driver = webdriver.Chrome()
    login(driver, username, password)
    driver.get("https://www.visuallocalization.net/mysubmissions/")

    rows_with_delete = driver.find_elements(
        By.XPATH, "//tr[td/a[contains(@href, '/delete/')]]"
    )

    data = {}
    for row in rows_with_delete:
        table = row.find_element(By.XPATH, "./ancestor::table")
        h3_element = table.find_element(By.XPATH, "./preceding::h3[1]")
        ds_id = h3_element.text
        numbers = extract_numbers(" ".join(row.text.split(" ")[1:]))
        if "sift" in row.text:
            str_ = row.text.replace("sift", "dog")
        else:
            str_ = row.text
        desc = get_present_substrings(str_)
        method_name_ = "_".join(desc)
        data.setdefault(ds_id, []).append([method_name_, numbers])
        print(method_name_)
    return data


def delete():
    driver = webdriver.Chrome()
    login(driver, USERNAME, PASSWORD)
    driver.get("https://www.visuallocalization.net/mysubmissions/")

    rows_with_delete = driver.find_elements(
        By.XPATH, "//tr[td/a[contains(@href, '/delete/')]]"
    )

    data = {}
    for row in rows_with_delete:
        delete_link = row.find_element(By.XPATH, ".//a[contains(@href, '/delete/')]")

        delete_href = delete_link.get_attribute("href")
        table = row.find_element(By.XPATH, "./ancestor::table")
        h3_element = table.find_element(By.XPATH, "./preceding::h3[1]")
        ds_id = h3_element.text
        data.setdefault(ds_id, []).append([row.text.split(" ")[0], delete_href])

    # for name, link in data["My Aachen Day-Night v1.1 Submissions"]:
    #     print(name)
    #     driver.get(link)
    #     WebDriverWait(driver, 10).until(EC.url_changes(link))

    for key1 in data:
        if key1 in ["My Aachen Day-Night Submissions"]:
            for name, link in data[key1]:
                print(name)
                driver.get(link)
                WebDriverWait(driver, 10).until(EC.url_changes(link))

    # for name, link in data["My RobotCar-Seasons v2 Submissions"]:
    #     if "sampler" in name.lower() or "results" in name.lower():
    #         print(name)
    #         driver.get(link)
    #         WebDriverWait(driver, 10).until(EC.url_changes(link))


def submit(submissions):
    # Set up your WebDriver (Chrome in this example)
    driver = webdriver.Chrome()

    login(driver, USERNAME, PASSWORD)

    # Open the benchmark submission page

    # Loop through each submission
    for dataset, file_path, method in submissions:
        driver.get(
            "https://www.visuallocalization.net/submission/"
        )  # Replace with the actual URL

        print(dataset, file_path, method)
        # Enter the dataset name
        dataset_input = driver.find_element(
            By.ID, "id_dataset"
        )  # Update with actual ID

        driver.execute_script(f"arguments[0].value = '{dataset}';", dataset_input)

        assert (
            dataset_input.get_attribute("value") == dataset
        ), "Input value does not match!"

        # Upload the file
        file_input = driver.find_element(
            By.ID, "id_result_file"
        )  # Update with actual ID
        file_input.send_keys(os.path.abspath(file_path))

        # Enter the method name
        method_input = driver.find_element(
            By.ID, "id_method_name"
        )  # Update with actual ID
        method_input.send_keys(method)

        # Submit the form
        submit_button = driver.find_element(By.XPATH, "//button[@type='submit']")

        submit_button.click()
        # # Wait for some time to allow submission to complete before next
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//p[text()=' Below you may view your submissions."
                    " If you want to edit one of your submissions,"
                    " or resubmit a new results file, "
                    "click on the edit link in the corresponding row to the right. ']",
                )
            )
        )

    # Close the browser
    driver.quit()


def is_file_empty(file_path):
    try:
        with open(file_path, "r") as file:
            # Check if file is empty by reading the first character
            return file.read(1) == ""
    except FileNotFoundError:
        print(f"File {file_path} does not exist.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Login to a website.")
    parser.add_argument("username", type=str, help="Your username")
    parser.add_argument("password", type=str, help="Your password")

    args = parser.parse_args()

    # Call the main function with parsed arguments
    USERNAME, PASSWORD = args.username, args.password
    # view(USERNAME, PASSWORD)
    delete()
    ds2id = {"aachen": "aachenv11", "robotcar": "robotcarv2", "cmu": "extended-cmu"}
    output_dir = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output"
    all_res_files = glob.glob(f"{output_dir}/*/*_eval_*.txt")
    all_res_files = sorted(all_res_files)
    all_fields = []
    for txt_name in all_res_files:
        ds_name = txt_name.split(output_dir)[-1].split("/")[1]
        dataset_id = ds2id[ds_name]
        if not is_file_empty(txt_name):
            method_name = txt_name.split("/")[-1].split(".txt")[0]
            all_fields.append([dataset_id, txt_name, method_name])
    submit(all_fields)
