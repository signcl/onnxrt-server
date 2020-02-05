# -*- coding: utf-8 -*-

# -- stdlib --
import types

# -- third party --
# -- own --

# -- code --


class CheckTypeFailed(Exception):
    def __init__(self):
        Exception.__init__(self)
        self.path = []

    def path_string(self):
        return ''.join([
            self._path_fragment(v)
            for v in self.path
        ])

    def _path_fragment(self, v):
        if isinstance(v, int):
            return '[%s]' % v
        elif isinstance(v, str):
            return '.%s' % v

    def finalize(self):
        self.args = (self.path_string(), )


def _check(cond):
    if not cond:
        raise CheckTypeFailed


def _check_isinstance(obj, cls):
    try:
        _check(isinstance(obj, cls))
    except TypeError as e:
        raise CheckTypeFailed from e


_check_key_not_exists = object()


def check_type_exc(pattern, obj, path=None):
    try:
        if isinstance(pattern, (list, tuple)):
            _check_isinstance(obj, (list, tuple))
            if len(pattern) == 2 and pattern[-1] is ...:
                cls = pattern[0]
                for i, v in enumerate(obj):
                    check_type_exc(cls, v, i)
            else:
                _check(len(pattern) == len(obj))
                for i, (cls, v) in enumerate(zip(pattern, obj)):
                    check_type_exc(cls, v, i)

        elif isinstance(pattern, dict):
            _check_isinstance(obj, dict)
            if ... in pattern:
                pattern = dict(pattern)
                match = pattern.pop(...)
            else:
                match = '!'

            if match in set('?!='):
                lkeys = set(pattern.keys())
                rkeys = set(obj.keys())

                if match == '!':
                    iterkeys = lkeys
                elif match == '?':
                    iterkeys = lkeys & rkeys
                elif match == '=':
                    _check(lkeys == rkeys)
                    iterkeys = lkeys
                else:
                    assert False, 'WTF?!'

                for k in iterkeys:
                    check_type_exc(pattern[k], obj.get(k, _check_key_not_exists), k)

            elif match is ...:
                assert len(pattern) == 1, 'Invalid dict pattern'
                kt, vt = list(pattern.items())[0]
                for k in obj:
                    check_type_exc(kt, k, '<%s>' % kt.__name__)
                    check_type_exc(vt, obj[k], k)
            else:
                assert False, 'Invalid dict match type'

        else:
            if issubclass(type(pattern), types.FunctionType):
                try:
                    _check(pattern(obj))
                except Exception as e:
                    raise CheckTypeFailed from e
            elif issubclass(type(pattern), (int, str, bytes, tuple)):
                _check(obj == pattern)
            else:
                _check_isinstance(obj, pattern)

    except CheckTypeFailed as e:
        if path is not None:
            e.path.insert(0, path)
        else:
            e.finalize()

        raise


def check_type(pattern, obj):
    try:
        check_type_exc(pattern, obj)
        return None
    except CheckTypeFailed as e:
        return e.path_string()
