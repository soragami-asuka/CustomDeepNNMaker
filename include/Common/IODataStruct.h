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

		U32 GetDataCount()const
		{
			return this->x * this->y * this->z * this->ch;
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

