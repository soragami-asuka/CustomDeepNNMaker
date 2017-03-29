//=======================================
// データフォーマットのベースインターフェース
//=======================================
#ifndef __GRAVISBELL_I_DATAFORMAT_BASE_H__
#define __GRAVISBELL_I_DATAFORMAT_BASE_H__

#include"../Common/Common.h"
#include"../Common/ErrorCode.h"
#include"../Common/IODataStruct.h"
#include"../Common/Guiddef.h"

#include"../SettingData/Standard/IData.h"

namespace Gravisbell {
namespace DataFormat {


	class IDataFormatBase
	{
	public:
		/** コンストラクタ */
		IDataFormatBase(){}
		/** デストラクタ */
		virtual ~IDataFormatBase(){}

	public:
		/** 名前の取得 */
		virtual const wchar_t* GetName()const = 0;
		/** 説明文の取得 */
		virtual const wchar_t* GetText()const = 0;

		/** X次元の要素数を取得 */
		virtual U32 GetBufferCountX(const wchar_t i_szCategory[])const = 0;
		/** Y次元の要素数を取得 */
		virtual U32 GetBufferCountY(const wchar_t i_szCategory[])const = 0;
		/** Z次元の要素数を取得 */
		virtual U32 GetBufferCountZ(const wchar_t i_szCategory[])const = 0;
		/** CH次元の要素数を取得 */
		virtual U32 GetBufferCountCH(const wchar_t i_szCategory[])const = 0;

		/** データ構造を取得 */
		virtual IODataStruct GetDataStruct(const wchar_t i_szCategory[])const = 0;

		/** カテゴリー数を取得する */
		virtual U32 GetCategoryCount()const = 0;
		/** カテゴリー名を番号指定で取得する */
		virtual const wchar_t* GetCategoryNameByNum(U32 categoryNo)const = 0;



	public:
		/** データをバイナリ形式で追加する.
			バイナリ形式はフォーマットの内容に関わらず[GetBufferCountZ()][GetBufferCountY()][GetBufferCountX()][GetBufferCountCH()]の配列データの先頭アドレスを渡す. */
//		virtual Gravisbell::ErrorCode AddDataByBinary(const F32 i_buffer[]) = 0;

		/** データ数を取得する */
		virtual U32 GetDataCount()const = 0;

		/** データを取得する */
		virtual const F32* GetDataByNum(U32 i_dataNo, const wchar_t i_szCategory[])const = 0;

	public:
		/** 正規化処理.
			データの追加が終了した後、一度のみ実行. 複数回実行すると値がおかしくなるので注意. */
		virtual Gravisbell::ErrorCode Normalize() = 0;
	};

}	// DataFormat
}	// Gravisbell



#endif	// __GRAVISBELL_I_DATAFORMAT_BASE_H__