from arrayqueues.shared_arrays import ArrayQueue, ArrayView
from multiprocessing import Process
import numpy as np
import pypylon.pylon as py


class MetaArrayQueue(ArrayQueue):
    """A small extension to support metadata saved alongside arrays"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def put(self, element, meta_data=None):

        if self.view is None or not self.view.fits(element):
            self.view = ArrayView(
                self.array.get_obj(), self.maxbytes, element.dtype, element.shape
            )
        else:
            self.check_full()

        qitem = self.view.push(element)

        self.queue.put((meta_data, qitem))

    def get(self, **kwargs):
        meta_data, aritem = self.queue.get(**kwargs)
        if self.view is None or not self.view.fits(aritem):
            self.view = ArrayView(self.array.get_obj(), self.maxbytes, *aritem)
        self.read_queue.put(aritem[2])
        return meta_data, self.view.pop(aritem[2])


class ImageProcess(Process):
    def __init__(self, source_queue):
        super().__init__()
        self.source_queue = source_queue

    def run(self):
        while True:
            meta, img = self.source_queue.get()
            print(np.mean(img), meta)


class CameraProcess(Process):
    def __init__(self, source_queue):
        super().__init__()
        self.source_queue = source_queue
        self.tlf = py.TlFactory.GetInstance()

    def run(self):
        cam = py.InstantCamera(self.tlf.CreateFirstDevice())
        cam.Open()

        cam.StartGrabbing()
        while True:
            with cam.RetrieveResult(1000) as res:
                if res.GrabSucceeded():
                    # one copy to get the frame into python memory
                    # this would be possible to skip, but then the release of the buffer
                    # would have to be synced with 'ImageProcess' beeing finished with processing of frame
                    img = res.Array
                    self.source_queue.put(img, {"timestamp": res.GetTimeStamp()})


if __name__ == "__main__":
    q = MetaArrayQueue(256)  # intitialises a MetaArrayQueue which can hold 256MB of data
    i_proc = ImageProcess(q)
    c_proc = CameraProcess(q)
    # start both processes
    i_proc.start()
    c_proc.start()
    # wait for completion ( in this example never )
    i_proc.join()
    c_proc.join()