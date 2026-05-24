from gui.app import App
import platform

if __name__ == "__main__":
    if platform.system() == 'Linux':
        app = App(video_source1=0, video_source2=2)
        app.mainloop()
    elif platform.system() == 'Windows':
        app = App(video_source1=0, video_source2=2)
        app.mainloop()
    else:
        print("Unsupported platform. Shutting down...")