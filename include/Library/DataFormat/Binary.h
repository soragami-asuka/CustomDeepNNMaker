//=====================================
// �f�[�^�t�H�[�}�b�g�̕�����z��`��
//=====================================
#ifndef __GRAVISBELL_LIBRARY_DATAFORMAT_BINARY_H__
#define __GRAVISBELL_LIBRARY_DATAFORMAT_BINARY_H__

#ifdef GRAVISBELL_LIBRAY_API
#undef GRAVISBELL_LIBRAY_API
#endif

#ifdef DATAFORMAT_BINARY_EXPORTS
#define GRAVISBELL_LIBRAY_API __declspec(dllexport)
#else
#define GRAVISBELL_LIBRAY_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#pragma comment(lib, "Gravisbell.DataFormat.Binary.lib")
#endif
#endif

#include"DataFormat/Binary/IDataFormat.h"



namespace Gravisbell {
namespace DataFormat {
namespace Binary {

	/** ������̔z���ǂݍ��ރf�[�^�t�H�[�}�b�g���쐬���� */
	extern GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormat(const wchar_t i_szName[], const wchar_t i_szText[]);

	/** ������̔z���ǂݍ��ރf�[�^�t�H�[�}�b�g���쐬���� */
	extern GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormatFromXML(const wchar_t szXMLFilePath[]);


}	// StringArray
}	// DataFormat
}	// Gravisbell


#endif	// __GRAVISBELL_DATAFORMAT_STRINGARRAY_H__