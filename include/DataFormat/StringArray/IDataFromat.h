//=======================================
// データフォーマットの文字列配列形式インターフェース
//=======================================
#ifndef __GRAVISBELL_I_DATAFORMAT_STRINGARRAY_H__
#define __GRAVISBELL_I_DATAFORMAT_STRINGARRAY_H__

#include"../IDataFormatBase.h"

namespace Gravisbell {
namespace DataFormat {
namespace StringArray {

	/** 正規化方法 */
	enum ENormalizeType
	{
		NORMALIZE_TYPE_NONE,				/**< 正規化なし */
		NORMALIZE_TYPE_MINMAX,				/**< 全データの最小値,最大値で正規化. */
		NORMALIZE_TYPE_VALUE,				/**< 指定された値で正規化する. */
		NORMALIZE_TYPE_AVERAGE_DEVIATION,	/**< 全データの平均値, 標準偏差を元に正規化 */
	};


	/** データフォーマット */
	class IDataFormat : public IDataFormatBase
	{
	public:
		/** コンストラクタ */
		IDataFormat(){}
		/** デストラクタ */
		virtual ~IDataFormat(){}

	public:
		/** データフォーマット数を取得する */
		virtual U32 GetDataFormatCount()const = 0;

		/** データフォーマットを全削除する */
		virtual Gravisbell::ErrorCode ClearDataFormat() = 0;

		//=============================================
		// float型
		//=============================================
		/** Float型データフォーマットを追加する. 正規化なし
			@param	i_szID			識別ID.
			@param	i_szCategory	データ種別. */
		virtual Gravisbell::ErrorCode AddDataFormatFloat(const wchar_t i_szID[], const wchar_t i_szCategory[]) = 0;

		/** Float型データフォーマットを追加する.
			全データの最小値、最大値で正規化
			@param	i_szID			識別ID.
			@param	i_szCategory	データ種別.
			@param	i_minOutput		出力される最小値.
			@param	i_maxOutput		出力される最大値. */
		virtual Gravisbell::ErrorCode AddDataFormatFloatNormalizeMinMax(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minOutput, F32 i_maxOutput) = 0;

		/** Float型データフォーマットを追加する.
			i_minValue, i_maxValue で正規化. 出力される値はi_minOutput, i_maxOutputの間になる.
			@param	i_szID			識別ID.
			@param	i_szCategory	データ種別.
			@param	i_minValue		データ内の最小値.
			@param	i_maxValue		データ内の最大値.
			@param	i_minOutput		出力される最小値.
			@param	i_maxOutput		出力される最大値. */
		virtual Gravisbell::ErrorCode AddDataFormatFloatNormalizeValue(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput) = 0;

		/** Float型データフォーマットを追加する.
			平均値と標準偏差を元に標準化する.
			加算平均-分散 が [i_minValue]
			加算平均+分散 が [i_maxValue]
			になるよう調整し、
			i_minValue -> i_minOutput
			i_maxValue -> i_maxOutput
			になるように正規化する
			@param	i_szID			識別ID.
			@param	i_szCategory	データ種別.
			@param	i_minValue		計算結果の最小値.
			@param	i_maxValue		計算結果の最大値.
			@param	i_minOutput		出力される最小値.
			@param	i_maxOutput		出力される最大値. */
		virtual Gravisbell::ErrorCode AddDataFormatFloatNormalizeAverageDeviation(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput) = 0;



		//=============================================
		// string型
		//=============================================
		/** string型データフォーマットを追加する. 正規化時に1,0の配列に変換する
			@param	i_szID			識別ID.
			@param	i_szCategory	データ種別. */
		virtual Gravisbell::ErrorCode AddDataFormatStringToBitArray(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minOutput, F32 i_maxOutput) = 0;
		/** string型データフォーマットを追加する. 正規化時にEnum値を元にした1,0の配列に変換する.
			@param	i_szID				識別ID.
			@param	i_szCategory		データ種別.
			@param	i_enumValueCount	enum値の数.
			@param	i_lpEnumString		enum値の文字列の配列.
			@param	i_defaultValue		入力データに所定の値が入っていなかった場合に設定されるデフォルト値. */
		virtual Gravisbell::ErrorCode AddDataFormatStringToBitArrayEnum(const wchar_t i_szID[], const wchar_t i_szCategory[], U32 i_enumDataCount, const wchar_t*const i_lpEnumData[], const wchar_t i_defaultData[], F32 i_minOutput, F32 i_maxOutput) = 0;


	public:
		/** データを文字列配列で追加する */
		virtual Gravisbell::ErrorCode AddDataByStringArray(const wchar_t*const i_szBuffer[]) = 0;
	};

}	// StringArray
}	// DataFormat
}	// Gravisbell



#endif // __GRAVISBELL_I_DATAFORMAT_CSV_H__