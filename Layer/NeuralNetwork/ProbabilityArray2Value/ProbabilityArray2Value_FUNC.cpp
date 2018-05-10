/*--------------------------------------------
 * FileName  : ProbabilityArray2Value_FUNC.cpp
 * LayerName : 確率の配列から値へ変換
 * guid      : 9E32D735-A29D-4636-A9CE-2C781BA7BE8E
 * 
 * Text      : 確率の配列から値へ変換する.
 *           : 最大値を取るCH番号を値に変換する.
 *           : 入力CH数＝分解能の整数倍である必要がある.
 *           : 学習時の入力に対する教師信号は正解信号を中心とした正規分布の平均値をとる
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard.h>

#include"ProbabilityArray2Value_FUNC.hpp"


// {9E32D735-A29D-4636-A9CE-2C781BA7BE8E}
static const Gravisbell::GUID g_guid(0x9e32d735, 0xa29d, 0x4636, 0xa9, 0xce, 0x2c, 0x78, 0x1b, 0xa7, 0xbe, 0x8e);

// VersionCode
static const Gravisbell::VersionCode g_version = {   1,   0,   0,   0}; 



struct StringData
{
    std::wstring name;
    std::wstring text;
};

namespace DefaultLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = L"ja";

    /** Base */
    static const StringData g_baseData = 
    {
        L"確率の配列から値へ変換",
        L"確率の配列から値へ変換する.\n最大値を取るCH番号を値に変換する.\n入力CH数＝分解能の整数倍である必要がある.\n学習時の入力に対する教師信号は正解信号を中心とした正規分布の平均値をとる"
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"outputMinValue",
            {
                L"出力最小値",
                L"",
            }
        },
        {
            L"outputMaxValue",
            {
                L"出力最大値",
                L"",
            }
        },
        {
            L"resolution",
            {
                L"分解能",
                L"",
            }
        },
        {
            L"variance",
            {
                L"教師信号の分散",
                L"",
            }
        },
        {
            L"allocationType",
            {
                L"割り当て種別",
                L"CH番号→値に変換するための変換方法",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"allocationType",
            {
                {
                    L"max",
                    {
                        L"最大値",
                        L"CH内の最大値を出力する",
                    },
                },
                {
                    L"average",
                    {
                        L"平均",
                        L"CH番号とCHの値を掛け合わせた値の平均値を出力する(相加平均)",
                    },
                },
            }
        },
    };



    /** ItemData Runtiime Parameter <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Runtime = 
    {
    };


    /** ItemData Runtime Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Runtime =
    {
    };


}


namespace CurrentLanguage
{
    /** Language Code */
    static const std::wstring g_languageCode = DefaultLanguage::g_languageCode;


    /** Base */
    static StringData g_baseData = DefaultLanguage::g_baseData;


    /** ItemData Layer Structure <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_LayerStructure = DefaultLanguage::g_lpItemData_LayerStructure;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure = DefaultLanguage::g_lpItemDataEnum_LayerStructure;


    /** ItemData Runtime <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Runtime;

    /** ItemData Runtime Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Runtime;

}



/** Acquire the layer identification code.
  * @param  o_layerCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)
{
	o_layerCode = g_guid;

	return Gravisbell::ERROR_CODE_NONE;
}

/** Get version code.
  * @param  o_versionCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode)
{
	o_versionCode = g_version;

	return Gravisbell::ERROR_CODE_NONE;
}


/** Create a layer structure setting.
  * @return If successful, new configuration information.
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSetting(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	/** Name : 出力最小値
	  * ID   : outputMinValue
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"outputMinValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMinValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMinValue"].text.c_str(),
			-32767.0000000000000000f, 32767.0000000000000000f, 0.0000000000000000f));

	/** Name : 出力最大値
	  * ID   : outputMaxValue
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"outputMaxValue",
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMaxValue"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"outputMaxValue"].text.c_str(),
			-32767.0000000000000000f, 32767.0000000000000000f, 0.0000000000000000f));

	/** Name : 分解能
	  * ID   : resolution
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"resolution",
			CurrentLanguage::g_lpItemData_LayerStructure[L"resolution"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"resolution"].text.c_str(),
			2, 65535, 2));

	/** Name : 教師信号の分散
	  * ID   : variance
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"variance",
			CurrentLanguage::g_lpItemData_LayerStructure[L"variance"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"variance"].text.c_str(),
			0.0000000000000000f, 65535.0000000000000000f, 1.0000000000000000f));

	/** Name : 割り当て種別
	  * ID   : allocationType
	  * Text : CH番号→値に変換するための変換方法
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"allocationType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"allocationType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"allocationType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"max",
			L"最大値",
			L"CH内の最大値を出力する");
		// 1
		pItemEnum->AddEnumItem(
			L"average",
			L"平均",
			L"CH番号とCHの値を掛け合わせた値の平均値を出力する(相加平均)");

pItemEnum->SetDefaultItem(0);
pItemEnum->SetValue(pItemEnum->GetDefault());

		pLayerConfig->AddItem(pItemEnum);
	}

	return pLayerConfig;
}

/** Create layer structure settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLayerStructureSetting();
	if(pLayerConfig == NULL)
		return NULL;

	Gravisbell::S64 useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


/** Create a runtime parameters.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameter(void)
{
	Gravisbell::GUID layerCode;
	GetLayerCode(layerCode);

	Gravisbell::VersionCode versionCode;
	GetVersionCode(versionCode);


	// Create Empty Setting Data
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = Gravisbell::SettingData::Standard::CreateEmptyData(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	// Create Item
	return pLayerConfig;
}

/** Create runtime parameter from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameterFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateRuntimeParameter();
	if(pLayerConfig == NULL)
		return NULL;

	Gravisbell::S64 useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


