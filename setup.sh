mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"mtejeda@bu.edu\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\\n\
[theme]\n\
base=\"light\"\n\
primaryColor=\"#036635\"\n\
secondaryBackgroundColor=\"#9a7a59\"\n\
textColor=\"#141313\"\n\
" > ~/.streamlit/config.toml
