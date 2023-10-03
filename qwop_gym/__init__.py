# =============================================================================
# Copyright 2023 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import gymnasium
from .envs.v1.qwop_env import QwopEnv
from .wrappers.verbose_wrapper import VerboseWrapper
from .wrappers.record_wrapper import RecordWrapper

all = [QwopEnv, VerboseWrapper, RecordWrapper]

gymnasium.register(id="QWOP-v1", entry_point="qwop_gym:QwopEnv")
