//=======================================
// 入出力データ構造
//=======================================
#ifndef __GRAVISBELL_IO_DATA_STRUCT__
#define __GRAVISBELL_IO_DATA_STRUCT__

#include"Common.h"

namespace Gravisbell {

	/** 入出力データの次元数 */
	const U32 IODATA_DIMENTION_COUNT = 3;

	/** 入出力データ構造 */
	struct IODataStruct
	{
		/** 1要素が持つデータ数 */
		U32 ch;
		/** 各軸ごとの要素数 */
		union
		{
			struct
			{
				U32 x;
				U32 y;
				U32 z;
			};
			U32 lpDim[IODATA_DIMENTION_COUNT];
		};

		IODataStruct(U32 ch=0, U32 x=1, U32 y=1, U32 z=1)
			:	x	(x)
			,	y	(y)
			,	z	(z)
			,	ch	(ch)
		{
		}

		bool operator==(const IODataStruct& dataStruct)const
		{
			if(this->x != dataStruct.x)
				return false;
			if(this->y != dataStruct.y)
				return false;
			if(this->z != dataStruct.z)
				return false;
			if(this->ch != dataStruct.ch)
				return false;

			return true;
		}
		bool operator!=(const IODataStruct& dataStruct)const
		{
			return !(*this == dataStruct);
		}

		U32 GetDataCount()const
		{
			return this->x * this->y * this->z * this->ch;
		}

		U32 POSITION_TO_OFFSET(U32 x, U32 y, U32 z, U32 ch)
		{
			return (((((ch*this->z+z)*this->y)+y)*this->x)+x);
		}
	};

}	// Gravisbell

#endif


/*

実際のデータ構造は
array[ch][z][y][x]

ch = 3
X=3,
Y=4,
Z=2,
{
  { {{0,1,2}, {0,1,2}, {0,1,2}},{0,1,2}}, }
  { {{0,1,2}, {0,1,2}, {0,1,2}},{0,1,2}}, }
}
{
  { {{0,1,2}, {0,1,2}, {0,1,2}},{0,1,2}}, }
  { {{0,1,2}, {0,1,2}, {0,1,2}},{0,1,2}}, }
},
{
  { {{0,1,2}, {0,1,2}, {0,1,2}},{0,1,2}}, }
  { {{0,1,2}, {0,1,2}, {0,1,2}},{0,1,2}}, }
},

*/

