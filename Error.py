# hold any kind of error.
# Zhang Ji, 20160514

class Error(RuntimeError):

    _traceback_ = []

    def __init__(self, ierr: int =0):
        self.ierr = ierr
        RuntimeError.__init__(self, self.ierr)

    def __nonzero__(self):
        return self.ierr != 0

    def __repr__(self):
        return 'StokesFlow.Error(%d)' % self.ierr

    def __str__(self):
        return 'StokesFlow.Error(%d)' % self.ierr
