import reflex as rx

config = rx.Config(
    app_name="smogcast_app",
    plugins=[rx.plugins.RadixThemesPlugin(theme=rx.theme(accent_color="teal"))],
    disable_plugins=[rx.plugins.SitemapPlugin],
    default_color_mode="dark",
    show_built_with_reflex=False,
)
