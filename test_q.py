import questionary
choices = [
    questionary.Choice(
        title=[("class:info", "Hello "), ("class:warning", "World")],
        value="1"
    ),
    questionary.Choice("Done", value="done")
]
style = questionary.Style([
    ("info", "fg:cyan"),
    ("warning", "fg:red")
])
questionary.select("Test", choices=choices, style=style).ask()
