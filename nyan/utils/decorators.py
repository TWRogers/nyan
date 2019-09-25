class History(object):
    def __init__(self, _type):
        assert _type in ('debug', 'transform')
        self._type = _type

    @classmethod
    def transform(cls):
        return cls(_type='transform')

    @classmethod
    def debug(cls):
        return cls(_type='debug')

    def __call__(self, f):
        f_name = f.__name__
        _type = self._type

        def required_f(*args, **kwargs):
            if _type == 'debug':
                raise NotImplementedError
            elif _type == 'transform':
                args[0]._transform_history.append({'fn_args': {'args': args[1:], 'kwargs': kwargs},
                                                   'fn_name': f_name,
                                                   'size': args[0].size,
                                                   'label': None})
            return f(*args, **kwargs)

        return required_f