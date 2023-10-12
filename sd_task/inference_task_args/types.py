from typing_extensions import Annotated
from pydantic.functional_validators import BeforeValidator
from pydantic.functional_serializers import PlainSerializer
from annotated_types import Gt, Le, MinLen


def fraction_int_to_float(fraction: int) -> float:
    if type(fraction) == 'int':
        return float(fraction) / 100.0
    else:
        return fraction


def float_to_fraction_int(fraction: float) -> int:
    return int(fraction * 100)


FloatFractionAsInt = Annotated[
    float,
    Gt(0),
    Le(1),
    BeforeValidator(fraction_int_to_float),
    PlainSerializer(float_to_fraction_int, return_type=int, when_used='json')
]

NonEmptyString = Annotated[
    str,
    MinLen(1)
]
