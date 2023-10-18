from typing import Optional
from functools import wraps, partial
import hashlib
import json
import pickle
from pickle import UnpicklingError
import redis


class RedisCache:
    redis_backend: Optional[redis.StrictRedis]
    bust_cache: bool
    disabled: bool

    keystore: dict

    def __init__(
            self,
            host: str = 'localhost',
            port: int = 6379,
            db: int = 0,
            bust_cache: bool = False,
            disabled: bool = False,
            *args,
            **kwargs
    ):
        self.keystore = {}
        if disabled:
            self.disable()
        else:
            self.enable(host, port, db, bust_cache, *args, **kwargs)

    def enable(
            self,
            host: str = 'localhost',
            port: int = 6379,
            db: int = 0,
            bust_cache: bool = False,
            *args,
            **kwargs
    ):
        self.redis_backend = None
        self.bust_cache = True
        self.disabled = True

        try:
            self.redis_backend = redis.StrictRedis(host=host, port=port, db=db)
            self.bust_cache = bust_cache
            self.disabled = False

            print("Redis server connected.")
        except Exception as e:

            print(
                f"WARNING: Redis could not connect to the database, cache disabled. ERROR: {e}"
            )

    def disable(self):
        self.redis_backend = None
        self.bust_cache = True
        self.disabled = True

    @staticmethod
    def make_hash(o):
        """
        Make a stable hash out of anything that is json serializable.  See json.JSONEncoder if you need to serialize
        a custom class.
        """
        obj_str = json.dumps(o, sort_keys=True)
        f_str_enc = obj_str.encode()

        m = hashlib.md5(f_str_enc)
        h = m.hexdigest()
        return h

    def cached(self, f=None, data_ex=None, no_data_ex=None, prepended_key_attr: str = None):

        """
        A cache decorator.
        If called without f, we've been called with optional arguments.
        We return a partial (decorator) with the optional args filled in.
        You can also call cached without specifying the optional args.
        :param f: the method to decorate
        :param data_ex: how long to cache the data in seconds. None means forever.
        :param no_data_ex: how long to cache no data in seconds (None, [], etc...). None means forever.
        :param prepended_key_attr: extra string to add to the computed hash that serves as a redis key
        :return: A wrapper function that performs caching
        """

        if f is None:
            return partial(self.cached, data_ex=data_ex, no_data_ex=no_data_ex, prepended_key_attr=prepended_key_attr)

        @wraps(f)
        def wrapper(*args, **kwargs):

            # If there are conditional values in the prepended key args that aren't satisfied, we do not cache.
            no_cache = False
            key = self._key(f, *args, **kwargs)
            if prepended_key_attr:
                prepended_key_attrs = prepended_key_attr.split(',')
                prepended_str = ''
                check_keystore = False
                for attr in prepended_key_attrs:
                    # conditional on if we only allow to cache attributes at specific values.
                    if '=' in attr:
                        name, val = attr.split('=')
                        attr = getattr(args[0], name)
                        prepended_str += f'{str(attr)}'
                        if attr != eval(val):
                            check_keystore = True
                    else:
                        prepended_str += f'{str(getattr(args[0], attr))}'
                key = f'{prepended_str}{key}'
                if check_keystore:
                    self.keystore[key] = self.keystore.get(key, -1) + 1
                    key = f'{key}.{self.keystore[key]}'

            # look in the cache unless we're busting the cache
            if not self.bust_cache and not self.disabled and not no_cache:
                if self.redis_backend.exists(key):

                    pickled = self.redis_backend.get(key)
                    try:
                        return pickle.loads(pickled)
                    except UnpicklingError:
                        pass

            # run the function
            v = f(*args, **kwargs)

            if not self.disabled and not no_cache:
                # pickle and cache the result
                pickled = pickle.dumps(v)

                if data_ex and v is not None:
                    ex = data_ex
                elif no_data_ex and v is None:
                    ex = no_data_ex
                else:
                    ex = None

                self.redis_backend.set(key, pickled, ex)

            # return the result
            return v

        return wrapper

    def _key(self, f, *args, **kwargs):
        func_name = f.__qualname__
        s = func_name

        # args[0] is the calling class, if there is one
        if len(args) > 1:
            for arg in args[1:]:

                arg_hash = self.make_hash(arg)
                s += "_{0}".format(arg_hash)

        elif len(args) == 1:
            # calling code is not in a named class
            arg_hash = self.make_hash(args[0])
            s += "_{0}".format(arg_hash)

        for k in sorted(kwargs.keys()):

            v = kwargs[k]
            v_hash = self.make_hash(v)
            s += "_{0}={1}".format(k, v_hash)

        return s
