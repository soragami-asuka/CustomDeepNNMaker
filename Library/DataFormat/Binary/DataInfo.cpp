//====================================
// データ本体の定義
//====================================
#include"stdafx.h"

#include"DataFormatClass.h"


namespace Gravisbell {
namespace DataFormat {
namespace Binary {


	/** コンストラクタ */
	DataInfo::DataInfo()
	{
	}
	/** デストラクタ */
	DataInfo::~DataInfo()
	{
	}

	/** データ数を取得 */
	U32 DataInfo::GetDataCount()const
	{
		return (U32)this->lpData.size();
	}

}	// Binary
}	// DataFormat
}	// Gravisbell