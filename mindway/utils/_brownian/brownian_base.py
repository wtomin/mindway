# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc


class BaseBrownian(metaclass=abc.ABCMeta):
    __slots__ = ()

    @abc.abstractmethod
    def __call__(self, ta, tb=None, return_U=False, return_A=False):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def shape(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def levy_area_approximation(self):
        raise NotImplementedError

    def size(self):
        return self.shape
