import streamlit as st
from script import gemini_answer, create_rules_list, inference
import time


default_sales_deck="""Welcome to BrightFuture Investments! We are dedicated to providing top-notch investment opportunities tailored to your financial goals. With our expert team and innovative strategies, your financial future is in safe hands. At BrightFuture Investments, we understand the complexities of the financial market and strive to simplify the investment process for you. Our mission is to help you achieve your financial aspirations with confidence and ease.
    BrightFuture Investments leverages cutting-edge algorithms and market insights to maximize your returns. Our team of experts has developed a sophisticated investment strategy that has historically delivered exceptional results. Many of our clients have seen their investments grow significantly, often doubling within a short period. While we always emphasize that past performance does not guarantee future results, our track record speaks volumes about our capability and dedication. Our focus on minimizing risk while maximizing returns sets us apart in the industry. Our platform consistently outperforms the competition, making it the preferred choice for savvy investors. We pride ourselves on our ability to deliver superior returns and unparalleled service. Many of our clients achieve their financial independence much faster than they anticipated, thanks to our innovative approach. By choosing BrightFuture Investments, you are aligning yourself with a team that prioritizes your financial success and is committed to helping you reach your goals.
    At BrightFuture Investments, we offer personalized investment plans tailored to your unique needs and objectives. Our comprehensive approach ensures that every aspect of your financial journey is carefully considered and optimized for maximum growth. From the initial consultation to ongoing portfolio management, we are with you every step of the way, providing expert guidance and support.
    Our advanced technology and analytical tools enable us to stay ahead of market trends and make informed investment decisions. This proactive approach allows us to capitalize on opportunities and mitigate risks effectively. Our clients benefit from our deep market knowledge and strategic insights, which are integral to achieving consistent and impressive returns.
    Moreover, we are committed to transparency and integrity in all our dealings. Our clients have access to detailed reports and updates on their investment performance, ensuring they are always informed and confident in their financial decisions. We believe in building long-term relationships based on trust and mutual success.
    In summary, BrightFuture Investments is your partner in achieving financial success. With our proven strategies, expert team, and commitment to excellence, you can rest assured that your investments are in capable hands. Join us today and take the first step towards a brighter financial future. Let us help you turn your financial dreams into reality with confidence and peace of mind.
    """

default_system_message="""You are a compliance officer.
    Your task is to understand the following rule and verify its adherence in the given sales deck.

    The steps are as follows:
    Understand the given rule.
    Augment the rule with additional vocabulary related to financial products.
    Evaluate the following sales deck: to determine if it respects the rule.

    Provide the output in JSON format with the following fields:
    rule_name (str): The name of the rule being applied.
    label (bool): The result of evaluating adherence to the rule.
    part (list[str]): Specific sections or aspects of the sales deck evaluated, including relevant details.
    suggestion (list[str]): Recommended changes or improvements to ensure compliance with the rule.

    Example JSON output structure:
    {
    "rule_name": "name of the rule",
    "label": "Result of evaluating adherence to the rule, either True or False",
    "part": ["Specific section or aspect of the sales pitch evaluated, including relevant details"],
    "suggestion": ["Recommended briefly some changes or improvements to ensure compliance with the rule, including relevant details"]
    }
    """

# Define the main function
def main():

    
    # Set the title of the app
    st.title('Poc: Prompt Testing & Enhancement')

    st.subheader('Model Selection')
    # Dropdown to select the model
    #model_name = st.selectbox("Select Model", ['gemini-1.5-flash', 'gemini-1.5-pro-latest', 'gemini-1.5-pro', 'gemini-1.5-flash-exp_0827', 'gemini-1.5-pro-exp_0827'])
    model_name = st.radio("Select Model", ['gemini-1.5-flash', 'gemini-1.5-pro-latest', 'gemini-1.5-pro', 'gemini-1.5-flash-exp_0827', 'gemini-1.5-pro-exp_0827'], horizontal=True)
    if model_name == 'gemini-1.5-flash':
        st.info("Rate limit: 15 Request Per Minute")
    if model_name == 'gemini-1.5-flash-exp_0827':
        st.info("Rate limit: 15 Request Per Minute")
    if model_name == 'gemini-1.5-pro-exp_0827':
        st.info("Rate limit: 15 Request Per Minute")
    if model_name == 'gemini-1.5-pro-latest':
        st.info("Rate limit: 2 Request Per Minute")
    if model_name == 'gemini-1.5-pro':
        st.info("Rate limit: 2 Request Per Minute")
    st.divider()
    st.subheader('Enter Sles Deck to evaluate here: ')
    sales_deck = st.text_area("Sales Deck:", value=default_sales_deck, height=250)

    # Input for rules
    st.divider()
    st.subheader('Enter rules here')
    st.write("\nIf you want to enter more than 1 rule use ## as a separator, \nexample:\n\n Fair, Clear, and Not Misleading ## Inclusion of Risk Warnings ")
    rules_string = st.text_area("Rules:", height=200)
    rules_list = create_rules_list(rules_string)

    st.divider()
    st.subheader('Prompt Engineering')
    st.write("Modify this prompt to evaluate the model output")
    # Input for prompt
    system_message = st.text_area("Prompt:", value=default_system_message, height=500)

    st.divider()
    st.subheader('Model Output')
    # Call the generate function
    generate_output = st.button('Generate output')
    if generate_output:
        start = time.time()
        with st.spinner(text="Generation In progress..."):
            output = inference(system_message, model_name, rules_list, sales_deck)
        end = time.time()
        st.write(f"Generation Duration: {end-start:.2f} seconds")
        for elm in output:
            st.json(elm)
    # Display the generated output in a text area
    #st.text_area("Generated Output:", value=output, height=250)

# Run the app
if __name__ == "__main__":
    main()
