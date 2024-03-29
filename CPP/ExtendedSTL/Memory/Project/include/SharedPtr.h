﻿#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#ifndef The
#define The (*this)
#endif // !The

#include "SharedPtr/BaseSharedPtr.h"
#include "SharedPtr/SharedPtrObj.h"

#if _HAS_CXX20
#include "SharedPtr/SharedPtrArr.h"
#else
#include "SharedPtr/SharedPtrPtr.h"
#endif

#include "SharedPtr/MakeShared.h"
#include "SharedPtr/ReferencePtr.h"

