// DataFormatStringArray.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"


#include"DataFormat.h"

#include<string>
#include<vector>
#include<list>
#include<set>
#include<map>

#include<boost/property_tree/ptree.hpp>
#include<boost/property_tree/xml_parser.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#include"Library/Common/StringUtility/StringUtility.h"


namespace Gravisbell {
namespace DataFormat {
namespace StringArray {
	
	struct BoolValue
	{
		F32 trueValue;	/**< trueの時の値 */
		F32 falseValue;	/**< falseの時の値 */

		BoolValue()
			:	trueValue	(1.0f)
			,	falseValue	(0.0f)
		{
		}
		BoolValue(F32 trueValue, F32 falseValue)
			:	trueValue	(trueValue)
			,	falseValue	(falseValue)
		{
		}
	};

	/** データフォーマットのアイテム */
	class CDataFormatItem
	{
	private:
		const std::wstring id;
		const std::wstring category;

	public:
		/** コンストラクタ */
		CDataFormatItem(const std::wstring& id, const std::wstring& category)
			:	id	(id)
			,	category	(category)
		{
		}
		/** デストラクタ */
		virtual ~CDataFormatItem(){}

	public:
		/** IDの取得 */
		const std::wstring& GetID()const{return this->id;}
		/** データ種別の取得 */
		const std::wstring& GetCategory()const{return this->category;}

		/** 使用バッファ数を返す */
		virtual U32 GetBufferCount()const = 0;

		/** バッファを取得する */
		virtual F32 GetBuffer(U32 dataNum, U32 bufferNum)const = 0;

		/** データを追加する */
		virtual ErrorCode AddData(const std::wstring& buf) = 0;

		/** 正規化 */
		virtual ErrorCode Normalize() = 0;
	};
	/** データフォーマットのアイテム.最小値,最大値指定あり */
	class CDataFormatItemMinMaxOutput
	{
	private:
		const F32 minOutput;
		const F32 maxOutput;

	public:
		/** コンストラクタ */
		CDataFormatItemMinMaxOutput(F32 i_minOutput, F32 i_maxOutput)
			:	minOutput		(i_minOutput)
			,	maxOutput		(i_maxOutput)
		{
		}
		/** デストラクタ */
		virtual ~CDataFormatItemMinMaxOutput(){}

	public:
		/** 出力値に変換する
			@param value	0～1の値. */
		F32 CalcOutputValue(F32 value){return min(maxOutput, max(minOutput, value * (maxOutput - minOutput) + minOutput));} 
	};

	/** データフォーマットのアイテム(float型) */
	class CDataFormatItemFloat : public CDataFormatItem
	{
	protected:
		std::list<F32> lpData;

	public:
		/** コンストラクタ */
		CDataFormatItemFloat(const std::wstring& id, const std::wstring& category)
			: CDataFormatItem(id, category)
		{
		}
		/** デストラクタ */
		virtual ~CDataFormatItemFloat()
		{
		}

	public:
		/** 使用バッファ数を返す */
		U32 GetBufferCount()const{return 1;}

		/** バッファを取得する */
		F32 GetBuffer(U32 dataNum, U32 bufferNum)const
		{
			if(dataNum >= this->lpData.size())
				return 0.0f;
			if(bufferNum >= 1)
				return 0.0f;

			auto it = this->lpData.begin();
			for(U32 i=0; i<dataNum; i++)
				it++;

			return *it;
		}

		/** データを追加する */
		ErrorCode AddData(const std::wstring& buf)
		{
			this->lpData.push_back((F32)_wtof(buf.c_str()));

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** 正規化 */
		virtual ErrorCode Normalize()
		{
			return ErrorCode::ERROR_CODE_NONE;
		}
	};
	/** データフォーマットのアイテム(float型)(最小値/最大値指定で正規化) */
	class CDataFormatItemFloatNormalizeMinMax : public CDataFormatItemFloat, public CDataFormatItemMinMaxOutput
	{
	public:
		CDataFormatItemFloatNormalizeMinMax(const std::wstring& id, const std::wstring& category, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItemFloat		(id, category)
			,	CDataFormatItemMinMaxOutput	(i_minOutput, i_maxOutput)
		{
		}
		virtual ~CDataFormatItemFloatNormalizeMinMax()
		{
		}
		
		/** 正規化 */
		virtual ErrorCode Normalize()
		{
			// 最小値と最大値を求める
			F32 minValue = -FLT_MAX;
			F32 maxValue =  FLT_MAX;
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					if(*it < minValue)
						minValue = *it;
					if(*it > maxValue)
						maxValue = *it;

					it++;
				}
			}

			// 最小値と最大値を用いて正規化
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					F32 value = (*it - minValue) / (maxValue - minValue);

					*it = this->CalcOutputValue(value);

					it++;
				}
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};
	/** データフォーマットのアイテム(float型)(指定された値の範囲で正規化する) */
	class CDataFormatItemFloatNormalizeValue : public CDataFormatItemFloat, public CDataFormatItemMinMaxOutput
	{
	private:
		F32 minValue;	/**< 最小値の値 */
		F32 maxValue;	/**< 最大値の値 */

	public:
		CDataFormatItemFloatNormalizeValue(const std::wstring& id, const std::wstring& category, F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItemFloat		(id, category)
			,	CDataFormatItemMinMaxOutput	(i_minOutput, i_maxOutput)
			,	minValue	(i_minValue)
			,	maxValue	(i_maxValue)
		{
		}
		virtual ~CDataFormatItemFloatNormalizeValue()
		{
		}

		/** 正規化 */
		virtual ErrorCode Normalize()
		{
			// 最小値と最大値を用いて正規化
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					F32 value = (*it - minValue) / (maxValue - minValue);

					*it = this->CalcOutputValue(value);

					it++;
				}
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};
	/** データフォーマットのアイテム(float型)(全データの平均値、標準偏差を元に標準化する) */
	class CDataFormatItemFloatNormalizeAverageDeviation : public CDataFormatItemFloat, public CDataFormatItemMinMaxOutput
	{
	private:
		F32 minValue;	/**< 最小値の値 */
		F32 maxValue;	/**< 最大値の値 */

	public:
		CDataFormatItemFloatNormalizeAverageDeviation(const std::wstring& id, const std::wstring& category, F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItemFloat		(id, category)
			,	CDataFormatItemMinMaxOutput	(i_minOutput, i_maxOutput)
			,	minValue	(i_minValue)
			,	maxValue	(i_maxValue)
		{
		}
		virtual ~CDataFormatItemFloatNormalizeAverageDeviation()
		{
		}

		/** 正規化 */
		virtual ErrorCode Normalize()
		{
			// 平均値を求める
			F32 average = 0.0f;
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					average += *it;
					it++;
				}
				average = average / this->lpData.size();
			}

			// 標準偏差
			F32 deviation = 0.0f;
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					deviation += (*it - average) * (*it - average);
					it++;
				}
				deviation = sqrt(deviation / this->lpData.size());
			}

			// 値を計算
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					F32 value = (*it - minValue) / (maxValue - minValue);

					*it = this->CalcOutputValue(value);

					it++;
				}
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** データフォーマットのアイテム(bool型) */
	class CDataFormatItemBool : public CDataFormatItem, public CDataFormatItemMinMaxOutput
	{
	protected:
		std::list<std::wstring> lpData;
		std::vector<F32> lpNormalizeValue;	/**< 正規化したデータ */

		std::set<std::wstring> lpFalseString;
		std::set<std::wstring> lpTrueString;

	public:
		CDataFormatItemBool(const std::wstring& id, const std::wstring& category, const std::vector<std::wstring>& i_lpFalseString, const std::vector<std::wstring>& i_lpTrueString, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItem				(id, category)
			,	CDataFormatItemMinMaxOutput	(i_minOutput, i_maxOutput)
		{
			for(auto buf : i_lpFalseString)
				this->lpFalseString.insert(buf);
			for(auto buf : i_lpTrueString)
				this->lpTrueString.insert(buf);
		}
		
	public:
		/** 使用バッファ数を返す */
		U32 GetBufferCount()const{return 1;}

		/** バッファを取得する */
		F32 GetBuffer(U32 dataNum, U32 bufferNum)const
		{
			if(dataNum >= this->lpNormalizeValue.size())
				return 0.0f;
			if(bufferNum >= 1)
				return 0.0f;

			return lpNormalizeValue[bufferNum];
		}

		/** データを追加する */
		ErrorCode AddData(const std::wstring& buf)
		{
			this->lpData.push_back(buf.c_str());

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** 正規化 */
		virtual ErrorCode Normalize()
		{
			this->lpNormalizeValue.clear();



		}
	};

	/** データフォーマットのアイテム(enum型) */
	class CDataFormatItemEnumBitArray : public CDataFormatItem, public CDataFormatItemMinMaxOutput
	{
	protected:
		std::list<std::wstring> lpData;
		std::list<std::vector<F32>> lpNormalizeValue;	/**< 正規化したデータ */

		std::vector<std::wstring> lpEnumData;
		std::wstring defaultData;

	public:
		CDataFormatItemEnumBitArray(const std::wstring& id, const std::wstring& category, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItem				(id, category)
			,	CDataFormatItemMinMaxOutput	(i_minOutput, i_maxOutput)
			,	defaultData	(L"")
		{
		}

	public:
		/** 使用バッファ数を返す */
		U32 GetBufferCount()const{return this->lpEnumData.size();}

		/** バッファを取得する */
		F32 GetBuffer(U32 dataNum, U32 bufferNum)const
		{
			if(dataNum >= this->lpNormalizeValue.size())
				return 0.0f;

			// データ位置を移動
			auto it_data = this->lpNormalizeValue.begin();
			for(U32 i=0; i<dataNum; i++)
				it_data++;

			// バッファ数を確認
			if(bufferNum >= it_data->size())
				return 0.0f;

			// バッファ位置を移動
			auto it_buf = it_data->begin();
			for(U32 i=0; i<bufferNum; i++)
				it_buf++;

			return *it_buf;
		}

		/** データを追加する */
		ErrorCode AddData(const std::wstring& buf)
		{
			this->lpData.push_back(buf.c_str());

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** 正規化 */
		virtual ErrorCode Normalize()
		{
			// 使用されている文字列を列挙する
			std::set<std::wstring> lpTmpEnumData;
			for(auto& data : this->lpData)
			{
				lpTmpEnumData.insert(data);
			}

			this->lpEnumData.clear();
			for(auto& data: lpTmpEnumData)
			{
				this->lpEnumData.push_back(data);
			}

			return this->NormalizeData();
		}
		/** 正規化 */
		ErrorCode NormalizeData()
		{
			// デフォルト文字列を値に変換
			U32 defaultValue = 0;
			{
				U32 value=0;
				for(auto enumValue : lpEnumData)
				{
					if(enumValue == defaultData)
					{
						defaultValue = value;
						break;
					}
					value++;
				}
			}

			// 値を正規化する
			this->lpNormalizeValue.clear();
			for(auto& data : this->lpData)
			{
				// 正規化後の値を保存するためのバッファを確保
				std::vector<F32> lpValue(this->lpEnumData.size());
				if(defaultValue < lpValue.size())
					lpValue[defaultValue] = this->CalcOutputValue(1.0f);

				// enum値の値を取得する
				U32 value=0;
				for(auto enumValue : this->lpEnumData)
				{
					if(enumValue == data)
					{
						lpValue[value] = this->CalcOutputValue(1.0f);
					}
					else
					{
						lpValue[value] = this->CalcOutputValue(0.0f);
					}
					value++;
				}

				// バッファを挿入
				this->lpNormalizeValue.push_back(lpValue);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};
	/** データフォーマットのアイテム(enum型) */
	class CDataFormatItemEnumBitArrayEnum : public CDataFormatItemEnumBitArray
	{
	public:
		CDataFormatItemEnumBitArrayEnum(const std::wstring& id, const std::wstring& category, const std::vector<std::wstring>& i_lpEnumString, std::wstring& i_defaultData, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItemEnumBitArray(id, category, i_minOutput, i_maxOutput)
		{
			this->defaultData = i_defaultData;
			this->lpEnumData = i_lpEnumString;
		}

		/** 正規化 */
		virtual ErrorCode Normalize()
		{
			return this->NormalizeData();
		}
	};

	/** データフォーマット */
	class CDataFormat : public IDataFormat
	{
	private:
		std::wstring name;	/**< 名前 */
		std::wstring text;	/**< 説明文 */

		std::set<std::wstring> lpCategoryName;	/**< データ種別名一覧 */

		U32 dataCount;	/**< データ数 */

		std::map<std::wstring, std::vector<F32>> lpTmpOutput;	/**< 出力データ格納用の一時バッファ */
		std::vector<CDataFormatItem*> lpDataFormat;	/**< データフォーマットの一覧 */

	public:
		/** コンストラクタ */
		CDataFormat()
		:	CDataFormat(L"", L"")
		{
		}
		/** コンストラクタ */
		CDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
		:	name		(i_szName)
		,	text		(i_szText)
		,	dataCount	(0)
		{
		}
		/** デストラクタ */
		virtual ~CDataFormat()
		{
			// データフォーマットを全削除
			this->ClearDataFormat();
		}

	public:
		/** 名前の取得 */
		const wchar_t* GetName()const
		{
			return name.c_str();
		}
		/** 説明文の取得 */
		const wchar_t* GetText()const
		{
			return text.c_str();
		}

		/** X次元の要素数を取得 */
		U32 GetBufferCountX(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** Y次元の要素数を取得 */
		U32 GetBufferCountY(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** Z次元の要素数を取得 */
		U32 GetBufferCountZ(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** CH次元の要素数を取得 */
		U32 GetBufferCountCH(const wchar_t i_szCategory[])const
		{
			U32 bufferCount = 0;
			for(auto it : this->lpDataFormat)
			{
				if(it == NULL)
					continue;

				// カテゴリをチェック
				if(it->GetCategory() != i_szCategory)
					continue;

				bufferCount += it->GetBufferCount();
			}

			return bufferCount;
		}

		/** データ構造を取得 */
		IODataStruct GetDataStruct(const wchar_t i_szCategory[])const
		{
			return IODataStruct(this->GetBufferCountCH(i_szCategory), this->GetBufferCountX(i_szCategory), this->GetBufferCountY(i_szCategory), this->GetBufferCountZ(i_szCategory));
		}

		/** カテゴリー数を取得する */
		U32 GetCategoryCount()const
		{
			return this->lpCategoryName.size();
		}
		/** カテゴリー名を番号指定で取得する */
		const wchar_t* GetCategoryNameByNum(U32 categoryNo)const
		{
			if(categoryNo >= this->lpCategoryName.size())
				return NULL;

			auto it = this->lpCategoryName.begin();
			for(U32 no=0; no<categoryNo; no++)
				it++;
			return it->c_str();
		}

	public:
		/** データ数を取得する */
		U32 GetDataCount()const
		{
			return this->dataCount;
		}

		/** データを取得する */
		const F32* GetDataByNum(U32 i_dataNo, const wchar_t i_szCategory[])const
		{
			if(i_dataNo >= this->dataCount)
				return NULL;

			auto it = this->lpTmpOutput.find(i_szCategory);
			if(it == this->lpTmpOutput.end())
				return NULL;

			return &it->second[0];
		}

	public:
		/** 正規化処理.
			データの追加が終了した後、一度のみ実行. 複数回実行すると値がおかしくなるので注意. */
		Gravisbell::ErrorCode Normalize()
		{
			for(auto it : this->lpDataFormat)
				it->Normalize();

			// 出力用一時バッファを確保
			this->lpTmpOutput.clear();
			for(auto& categoryName : this->lpCategoryName)
			{
				this->lpTmpOutput[categoryName].resize(this->GetBufferCountCH(categoryName.c_str()));
			}

			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		/** データフォーマット数を取得する */
		U32 GetDataFormatCount()const
		{
			return this->lpDataFormat.size();
		}

		/** データフォーマットを全削除する */
		Gravisbell::ErrorCode ClearDataFormat()
		{
			for(auto it : this->lpDataFormat)
			{
				delete it;
			}
			this->lpDataFormat.clear();

			return ErrorCode::ERROR_CODE_NONE;
		}

		//=============================================
		// float型
		//=============================================
		/** Float型データフォーマットを追加する. 正規化なし
			@param	i_szID			識別ID.
			@param	i_szCategory	データ種別. */
		Gravisbell::ErrorCode AddDataFormatFloat(const wchar_t i_szID[], const wchar_t i_szCategory[])
		{
			this->lpDataFormat.push_back(new CDataFormatItemFloat(i_szID, i_szCategory));

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** Float型データフォーマットを追加する.
			全データの最小値、最大値で正規化
			@param	i_szID			識別ID.
			@param	i_szCategory	データ種別.
			@param	i_minOutput		出力される最小値.
			@param	i_maxOutput		出力される最大値. */
		Gravisbell::ErrorCode AddDataFormatFloatNormalizeMinMax(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minOutput, F32 i_maxOutput)
		{
			this->lpDataFormat.push_back(new CDataFormatItemFloatNormalizeMinMax(i_szID, i_szCategory, i_minOutput, i_maxOutput));

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** Float型データフォーマットを追加する.
			i_minValue, i_maxValue で正規化. 出力される値はi_minOutput, i_maxOutputの間になる.
			@param	i_szID			識別ID.
			@param	i_szCategory	データ種別.
			@param	i_minValue		データ内の最小値.
			@param	i_maxValue		データ内の最大値.
			@param	i_minOutput		出力される最小値.
			@param	i_maxOutput		出力される最大値. */
		Gravisbell::ErrorCode AddDataFormatFloatNormalizeValue(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput)
		{
			this->lpDataFormat.push_back(new CDataFormatItemFloatNormalizeValue(i_szID, i_szCategory, i_minValue, i_maxValue, i_minOutput, i_maxOutput));

			return ErrorCode::ERROR_CODE_NONE;
		}

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
		Gravisbell::ErrorCode AddDataFormatFloatNormalizeAverageDeviation(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput)
		{
			this->lpDataFormat.push_back(new CDataFormatItemFloatNormalizeAverageDeviation(i_szID, i_szCategory, i_minValue, i_maxValue, i_minOutput, i_maxOutput));

			return ErrorCode::ERROR_CODE_NONE;
		}


		//=============================================
		// string型
		//=============================================
		/** string型データフォーマットを追加する. 正規化時に1,0の配列に変換する
			@param	i_szID			識別ID.
			@param	i_szCategory	データ種別. */
		Gravisbell::ErrorCode AddDataFormatStringToBitArray(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minOutput, F32 i_maxOutput)
		{
			this->lpDataFormat.push_back(new CDataFormatItemEnumBitArray(i_szID, i_szCategory, i_minOutput, i_maxOutput));

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** string型データフォーマットを追加する. 正規化時にEnum値を元にした1,0の配列に変換する.
			@param	i_szID				識別ID.
			@param	i_szCategory		データ種別.
			@param	i_enumValueCount	enum値の数.
			@param	i_lpEnumString		enum値の文字列の配列.
			@param	i_defaultValue		入力データに所定の値が入っていなかった場合に設定されるデフォルト値. */
		Gravisbell::ErrorCode AddDataFormatStringToBitArrayEnum(const wchar_t i_szID[], const wchar_t i_szCategory[], U32 i_enumDataCount, const wchar_t*const i_lpEnumData[], const wchar_t i_defaultData[], F32 i_minOutput, F32 i_maxOutput)
		{
			std::vector<std::wstring> lpEnumData;
			for(U32 i=0; i<i_enumDataCount; i++)
				lpEnumData.push_back(i_lpEnumData[i]);

			this->lpDataFormat.push_back(new CDataFormatItemEnumBitArrayEnum(i_szID, i_szCategory, lpEnumData, (std::wstring)i_defaultData, i_minOutput, i_maxOutput));

			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		/** データを文字列配列で追加する */
		Gravisbell::ErrorCode AddDataByStringArray(const wchar_t*const i_szBuffer[])
		{
			for(U32 i=0; i<this->lpDataFormat.size(); i++)
			{
				this->lpDataFormat[i]->AddData(i_szBuffer[i]);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** 文字列の配列を読み込むデータフォーマットを作成する */
	GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
	{
		return new CDataFormat(i_szName, i_szText);
	}
	/** 文字列の配列を読み込むデータフォーマットを作成する */
	extern GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormatFromXML(const wchar_t szXMLFilePath[])
	{
		using namespace StringUtility;

		// XMLファイルの読み込み
		boost::property_tree::ptree pXmlTree;
		boost::iostreams::file_descriptor_source fs(UnicodeToShiftjis(szXMLFilePath));
		boost::iostreams::stream<boost::iostreams::file_descriptor_source> fsstream(fs);
		try
		{
			boost::property_tree::read_xml(fsstream, pXmlTree);
		}
		catch(boost::exception& e)
		{
			e;
			return NULL;
		}

		CDataFormat* pDataFormat = NULL;
		try
		{
			// 名前
			std::wstring name;
			if(boost::optional<std::string> pValue = pXmlTree.get_optional<std::string>("DataFormat.Name"))
			{
				name = UTF8toUnicode(pValue.get());
			}
			// テキスト
			std::wstring text;
			if(boost::optional<std::string> pValue = pXmlTree.get_optional<std::string>("DataFormat.Text"))
			{
				text = UTF8toUnicode(pValue.get());
			}

			// bool値の値
			std::map<std::wstring, BoolValue>	lpBoolValue;	/**< bool値をF32に変換する設定値の一覧.	<データ種別名, 変換データ> */
			lpBoolValue[L""] = BoolValue();
			for(const boost::property_tree::ptree::value_type &it : pXmlTree.get_child("DataFormat.BoolValue"))
			{
				if(it.first == "true" || it.first == "false")
				{
					// 属性からカテゴリを取得
					std::wstring category = L"";
					if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.category"))
					{
						category = UTF8toUnicode(pValue.get());
					}

					if(it.first == "true")
						lpBoolValue[category].trueValue = (F32)atof(it.second.data().c_str());
					else if(it.first == "false")
						lpBoolValue[category].falseValue = (F32)atof(it.second.data().c_str());
				}
			}


			// データフォーマットを作成
			pDataFormat = new CDataFormat();


			// Channelの読み込み
			for(const boost::property_tree::ptree::value_type &it : pXmlTree.get_child("DataFormat.Channel"))
			{
				// idの取得
				std::wstring id = L"";
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.id"))
				{
					id = UTF8toUnicode(pValue.get());
				}
				// categoryの取得
				std::wstring category = L"";
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.category"))
				{
					category = UTF8toUnicode(pValue.get());
				}

				// bool型の値を取得
				BoolValue boolValue = lpBoolValue[L""];
				if(lpBoolValue.count(category))
					boolValue = lpBoolValue[category];

				if(it.first == "String")
				{
					enum UseType
					{
						USETYPE_BITARRAY,
						USETYPE_BITARRAY_ENUM
					};
					UseType useType = USETYPE_BITARRAY;

					// 使用方法を取得
					if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.useType"))
					{
						if(pValue.get() == "bit_array")
							useType = USETYPE_BITARRAY;
						else if(pValue.get() == "bit_array_enum")
							useType = USETYPE_BITARRAY_ENUM;
					}

					switch(useType)
					{
					case USETYPE_BITARRAY:
					default:
						{
							// フォーマットを追加
							pDataFormat->AddDataFormatStringToBitArray(id.c_str(), category.c_str(), boolValue.falseValue, boolValue.trueValue); 
						}
						break;
					case USETYPE_BITARRAY_ENUM:
						{
							// enum値を列挙
							std::list<std::wstring> lpEnumString;
							std::vector<const wchar_t*> lpEnumStringPointer;
							std::wstring defaultString = L"";
							if(auto& pTreeEnum = it.second.get_child_optional("Enum"))
							{
								for(const boost::property_tree::ptree::value_type &it_enum : pTreeEnum.get().get_child(""))
								{
									if(it_enum.first == "item")
									{
										lpEnumString.push_back(UTF8toUnicode(it_enum.second.data()));
										lpEnumStringPointer.push_back(lpEnumString.rbegin()->c_str());

										if(boost::optional<std::string> pValue = it_enum.second.get_optional<std::string>("<xmlattr>.default"))
										{
											if(pValue.get() == "true")
												defaultString = UTF8toUnicode(it_enum.second.data());
										}
									}
								}
							}

							// フォーマットを追加
							pDataFormat->AddDataFormatStringToBitArrayEnum(id.c_str(), category.c_str(), lpEnumStringPointer.size(), &lpEnumStringPointer[0], defaultString.c_str(), boolValue.falseValue, boolValue.trueValue); 
						}
						break;
					}
				}
				else if(it.first == "Float")
				{
					enum NormalizeType
					{
						NORMALIZETYPE_NONE,		// 正規化しない
						NORMALIZETYPE_MINMAX,	// 全データの最小値、最大値を元に正規化する
						NORMALIZETYPE_VALUE,	// 最小値、最大値を指定して正規化する
						NORMALIZETYPE_SDEV,		// 全データの平均値、標準偏差を元に正規化する
					};
					NormalizeType normalizeType = NORMALIZETYPE_NONE;

					// 正規化種別を取得
					if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.normalize"))
					{
						if(pValue.get() == "none")
							normalizeType = NORMALIZETYPE_NONE;
						else if(pValue.get() == "min_max")
							normalizeType = NORMALIZETYPE_MINMAX;
						else if(pValue.get() == "value")
							normalizeType = NORMALIZETYPE_VALUE;
						else if(pValue.get() == "average_deviation")
							normalizeType = NORMALIZETYPE_SDEV;
					}

					// 設定最小値, 最大値を取得する
					F32 minValue = 0.0f;
					F32 maxValue = 1.0f;
					if(boost::optional<float> pValue = it.second.get_optional<float>("min"))
						minValue = pValue.get();
					if(boost::optional<float> pValue = it.second.get_optional<float>("max"))
						maxValue = pValue.get();

					// 出力最小値、最大値を取得する
					if(boost::optional<float> pValue = it.second.get_optional<float>("output_min"))
						boolValue.falseValue = pValue.get();
					if(boost::optional<float> pValue = it.second.get_optional<float>("output_max"))
						boolValue.trueValue = pValue.get();

					switch(normalizeType)
					{
					case NORMALIZETYPE_NONE:		// 正規化しない
					default:
						pDataFormat->AddDataFormatFloat(id.c_str(), category.c_str());
						break;
					case NORMALIZETYPE_MINMAX:	// 全データの最小値、最大値を元に正規化する
						pDataFormat->AddDataFormatFloatNormalizeMinMax(id.c_str(), category.c_str(), boolValue.falseValue, boolValue.trueValue);
						break;
					case NORMALIZETYPE_VALUE:	// 最小値、最大値を指定して正規化する
						pDataFormat->AddDataFormatFloatNormalizeValue(id.c_str(), category.c_str(), minValue, maxValue, boolValue.falseValue, boolValue.trueValue);
						break;
					case NORMALIZETYPE_SDEV:		// 全データの平均値、標準偏差を元に正規化する
						pDataFormat->AddDataFormatFloatNormalizeAverageDeviation(id.c_str(), category.c_str(), minValue, maxValue, boolValue.falseValue, boolValue.trueValue);
						break;
					}
				}
			}
		}
		catch(boost::exception& e)
		{
			e;
			if(pDataFormat)
				delete pDataFormat;
			return NULL;
		}

		return pDataFormat;
	}


}	// StringArray
}	// DataFormat
}	// Gravisbell


