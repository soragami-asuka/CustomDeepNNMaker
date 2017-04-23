/*--------------------------------------------
 * FileName  : Convolution_FUNC.cpp
 * LayerName : 畳みこみニューラルネットワーク
 * guid      : F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA
 * 
 * Text      : 畳みこみニューラルネットワーク.
--------------------------------------------*/
#include"stdafx.h"

#include<string>
#include<map>

#include<Library/SettingData/Standard/SettingData.h>

#include"Convolution_FUNC.hpp"


// {F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA}
static const Gravisbell::GUID g_guid(0xf6662e0e, 0x1ca4, 0x4d59, 0xac, 0xca, 0xca, 0xc2, 0x9a, 0x16, 0xc0, 0xaa);

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
        L"畳みこみニューラルネットワーク",
        L"畳みこみニューラルネットワーク."
    };


    /** ItemData Layer Structure <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_LayerStructure = 
    {
        {
            L"Output_Channel",
            {
                L"出力チャンネル数",
                L"出力されるチャンネルの数",
            }
        },
        {
            L"DropOut",
            {
                L"ドロップアウト率",
                L"前レイヤーを無視する割合.\n1.0で前レイヤーの全出力を無視する",
            }
        },
        {
            L"FilterSize",
            {
                L"フィルタサイズ",
                L"畳みこみを行う入力信号数",
            }
        },
        {
            L"Move",
            {
                L"フィルタ移動量",
                L"1ニューロンごとに移動する入力信号の移動量",
            }
        },
        {
            L"Stride",
            {
                L"畳みこみ移動量",
                L"畳みこみごとに移動する入力信号の移動量",
            }
        },
        {
            L"PaddingM",
            {
                L"パディングサイズ(-方向)",
                L"",
            }
        },
        {
            L"PaddingP",
            {
                L"パディングサイズ(+方向)",
                L"",
            }
        },
        {
            L"PaddingType",
            {
                L"パディング種別",
                L"パディングを行う際の方法設定",
            }
        },
    };


    /** ItemData Layer Structure Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_LayerStructure =
    {
        {
            L"PaddingType",
            {
                {
                    L"zero",
                    {
                        L"ゼロパディング",
                        L"不足分を0で埋める",
                    },
                },
                {
                    L"border",
                    {
                        L"境界値",
                        L"不足分と隣接する値を参照する",
                    },
                },
                {
                    L"mirror",
                    {
                        L"反転",
                        L"不足分と隣接する値から逆方向に参照する",
                    },
                },
                {
                    L"clamp",
                    {
                        L"クランプ",
                        L"不足分の反対側の境目から順方向に参照する",
                    },
                },
            }
        },
    };



    /** ItemData Learn <id, StringData> */
    static const std::map<std::wstring, StringData> g_lpItemData_Learn = 
    {
        {
            L"LearnCoeff",
            {
                L"学習係数",
                L"",
            }
        },
    };


    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn =
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


    /** ItemData Learn <id, StringData> */
    static std::map<std::wstring, StringData> g_lpItemData_Learn = DefaultLanguage::g_lpItemData_Learn;

    /** ItemData Learn Enum <id, enumID, StringData> */
    static const std::map<std::wstring, std::map<std::wstring, StringData>> g_lpItemDataEnum_Learn = DefaultLanguage::g_lpItemDataEnum_Learn;

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
	/** Name : 出力チャンネル数
	  * ID   : Output_Channel
	  * Text : 出力されるチャンネルの数
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Int(
			L"Output_Channel",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Output_Channel"].text.c_str(),
			1, 65535, 1));

	/** Name : ドロップアウト率
	  * ID   : DropOut
	  * Text : 前レイヤーを無視する割合.
	  *      : 1.0で前レイヤーの全出力を無視する
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"DropOut",
			CurrentLanguage::g_lpItemData_LayerStructure[L"DropOut"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"DropOut"].text.c_str(),
			0.000000f, 1.000000f, 0.000000f));

	/** Name : フィルタサイズ
	  * ID   : FilterSize
	  * Text : 畳みこみを行う入力信号数
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"FilterSize",
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"FilterSize"].text.c_str(),
			1, 1, 1,
			65535, 65535, 65535,
			1, 1, 1));

	/** Name : フィルタ移動量
	  * ID   : Move
	  * Text : 1ニューロンごとに移動する入力信号の移動量
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Float(
			L"Move",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Move"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Move"].text.c_str(),
			0.100000f, 0.100000f, 0.100000f,
			65535.000000f, 65535.000000f, 65535.000000f,
			1.000000f, 1.000000f, 1.000000f));

	/** Name : 畳みこみ移動量
	  * ID   : Stride
	  * Text : 畳みこみごとに移動する入力信号の移動量
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Float(
			L"Stride",
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"Stride"].text.c_str(),
			0.100000f, 0.100000f, 0.100000f,
			65535.000000f, 65535.000000f, 65535.000000f,
			1.000000f, 1.000000f, 1.000000f));

	/** Name : パディングサイズ(-方向)
	  * ID   : PaddingM
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"PaddingM",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingM"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingM"].text.c_str(),
			0, 0, 0,
			65535, 65535, 65535,
			0, 0, 0));

	/** Name : パディングサイズ(+方向)
	  * ID   : PaddingP
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Vector3D_Int(
			L"PaddingP",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingP"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingP"].text.c_str(),
			0, 0, 0,
			65535, 65535, 65535,
			0, 0, 0));

	/** Name : パディング種別
	  * ID   : PaddingType
	  * Text : パディングを行う際の方法設定
	  */
	{
		Gravisbell::SettingData::Standard::IItemEx_Enum* pItemEnum = Gravisbell::SettingData::Standard::CreateItem_Enum(
			L"PaddingType",
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].name.c_str(),
			CurrentLanguage::g_lpItemData_LayerStructure[L"PaddingType"].text.c_str());

		// 0
		pItemEnum->AddEnumItem(
			L"zero",
			L"ゼロパディング",
			L"不足分を0で埋める");
		// 1
		pItemEnum->AddEnumItem(
			L"border",
			L"境界値",
			L"不足分と隣接する値を参照する");
		// 2
		pItemEnum->AddEnumItem(
			L"mirror",
			L"反転",
			L"不足分と隣接する値から逆方向に参照する");
		// 3
		pItemEnum->AddEnumItem(
			L"clamp",
			L"クランプ",
			L"不足分の反対側の境目から順方向に参照する");

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
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLayerStructureSetting();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


/** Create a learning setting.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSetting(void)
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
	/** Name : 学習係数
	  * ID   : LearnCoeff
	  */
	pLayerConfig->AddItem(
		Gravisbell::SettingData::Standard::CreateItem_Float(
			L"LearnCoeff",
			CurrentLanguage::g_lpItemData_Learn[L"LearnCoeff"].name.c_str(),
			CurrentLanguage::g_lpItemData_Learn[L"LearnCoeff"].text.c_str(),
			0.000000f, 1000.000000f, 1.000000f));

	return pLayerConfig;
}

/** Create learning settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	Gravisbell::SettingData::Standard::IDataEx* pLayerConfig = (Gravisbell::SettingData::Standard::IDataEx*)CreateLearningSetting();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}


