//=======================================
// データフォーマットのバイナリ形式インターフェース
//=======================================
#ifndef __GRAVISBELL_I_DATAFORMAT_BINARY_H__
#define __GRAVISBELL_I_DATAFORMAT_BINARY_H__

#include"../IDataFormatBase.h"

namespace Gravisbell {
namespace DataFormat {
namespace Binary {

	/** データフォーマット */
	class IDataFormat : public IDataFormatBase
	{
	public:
		/** コンストラクタ */
		IDataFormat(){}
		/** デストラクタ */
		virtual ~IDataFormat(){}

	public:
		/** ID指定で変数の値を取得する.(直値判定付き) */
		virtual S32 GetVariableValue(const wchar_t i_szID[])const = 0;
		/** ID指定で変数の値を取得する.(直値判定付き.)(float型として値を返す) */
		virtual F32 GetVariableValueAsFloat(const wchar_t i_szID[])const = 0;

	public:
		/** データをバイナリから読み込む.
			@param	i_lpBuf		バイナリデータ.
			@param	i_byteCount	使用可能なバイト数.	
			@return	実際に読み込んだバイト数. 失敗した場合は負の値 */
		virtual S32 LoadBinary(const BYTE i_lpBuf[], U32 i_byteCount) = 0;
	};

}	// Binary
}	// DataFormat
}	// Gravisbell



#endif // __GRAVISBELL_I_DATAFORMAT_CSV_H__