//====================================
// �f�[�^�t�H�[�}�b�g��`�̖{�̏��
//====================================
#ifndef __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_CLASS_H__
#define __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_CLASS_H__


#include "stdafx.h"


#include"Library/DataFormat/Binary.h"

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

#include"DataFormatItem.h"

namespace Gravisbell {
namespace DataFormat {
namespace Binary {

	class CDataFormat;

	/** �������l�ɕϊ�.�i������t�� */
	S32 ConvertString2Int(const std::wstring& buf);
	/** �������l�ɕϊ�.�i������t�� */
	U32 ConvertString2UInt(const std::wstring& buf);
	/** �������l�ɕϊ�.�i������t�� */
	F32 ConvertString2Float(const std::wstring& buf);

	/** �f�[�^�\�� */
	struct DataStruct
	{
		std::wstring m_x;
		std::wstring m_y;
		std::wstring m_z;
		std::wstring m_ch;

		F32 m_false;
		F32 m_true;
	};
	/** �f�[�^�{�� */
	struct DataInfo
	{
		DataStruct dataStruct;	/**< �f�[�^�\�� */
		std::vector<F32*> lpData;	/**< ���f�[�^ */

		/** �R���X�g���N�^ */
		DataInfo();
		/** �f�X�g���N�^ */
		~DataInfo();

		/** �f�[�^�����擾 */
		U32 GetDataCount()const;
	};


	/** �f�[�^�t�H�[�}�b�g */
	class CDataFormat : public IDataFormat
	{
	private:
		std::wstring name;	/**< ���O */
		std::wstring text;	/**< ������ */

		std::map<std::wstring, S32> lpVariable;	/**< �ϐ��ƌ��ݕϐ�������Ă���l. */
		std::map<std::wstring, DataInfo> lpData;			/**< �f�[�^�ꗗ */

		std::list<Format::CItem_base*> lpDataFormat;

		bool onReverseByteOrder;

	public:
		/** �R���X�g���N�^ */
		CDataFormat();
		/** �R���X�g���N�^ */
		CDataFormat(const wchar_t i_szName[], const wchar_t i_szText[], bool onReverseByteOrder);
		/** �f�X�g���N�^ */
		virtual ~CDataFormat();

	public:
		/** ���O�̎擾 */
		const wchar_t* GetName()const;
		/** �������̎擾 */
		const wchar_t* GetText()const;

		/** X�����̗v�f�����擾 */
		U32 GetBufferCountX(const wchar_t i_szCategory[])const;
		/** Y�����̗v�f�����擾 */
		U32 GetBufferCountY(const wchar_t i_szCategory[])const;
		/** Z�����̗v�f�����擾 */
		U32 GetBufferCountZ(const wchar_t i_szCategory[])const;
		/** CH�����̗v�f�����擾 */
		U32 GetBufferCountCH(const wchar_t i_szCategory[])const;

		/** �f�[�^�\�����擾 */
		IODataStruct GetDataStruct(const wchar_t i_szCategory[])const;

		/** �f�[�^�����擾 */
		const DataInfo* GetDataInfo(const wchar_t i_szCategory[])const;
		/** �f�[�^�����擾 */
		DataInfo* GetDataInfo(const wchar_t i_szCategory[]);

		/** �f�[�^����ǉ����� */
		Gravisbell::ErrorCode AddDataInfo(const wchar_t i_szCategory[], const wchar_t i_x[], const wchar_t i_y[], const wchar_t i_z[], const wchar_t i_ch[], F32 i_false, F32 i_true);
		/** �f�[�^���ɒl���������� */
		Gravisbell::ErrorCode SetDataValue(const wchar_t i_szCategory[], U32 i_No, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, F32 value);
		/** �f�[�^���ɒl����������.
			0.0=false.
			1.0=true.
			�Ƃ��Ēl���i�[���� */
		Gravisbell::ErrorCode SetDataValueNormalize(const wchar_t i_szCategory[], U32 i_No, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, F32 value);
		/** �f�[�^���ɒl���������� */
		Gravisbell::ErrorCode SetDataValue(const wchar_t i_szCategory[], U32 i_No, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, bool value);

		/** ID�w��ŕϐ��̒l���擾����.(���l����t��) */
		S32 GetVariableValue(const wchar_t i_szID[])const;
		/** ID�w��ŕϐ��̒l���擾����.(���l����t��.)(float�^�Ƃ��Ēl��Ԃ�) */
		F32 GetVariableValueAsFloat(const wchar_t i_szID[])const;
		/** ID�w��ŕϐ��ɒl��ݒ肷��.(���l����t��) */
		void SetVariableValue(const wchar_t i_szID[], S32 value);

		/** �J�e�S���[�����擾���� */
		U32 GetCategoryCount()const;
		/** �J�e�S���[����ԍ��w��Ŏ擾���� */
		const wchar_t* GetCategoryNameByNum(U32 categoryNo)const;

		/** Byte-Order�̔��]�t���O���擾���� */
		bool GetOnReverseByteOrder()const;

	public:
		/** �f�[�^�����擾���� */
		U32 GetDataCount()const;

		/** �f�[�^���擾���� */
		const F32* GetDataByNum(U32 i_dataNo, const wchar_t i_szCategory[])const;

	public:
		/** ���K������.
			�f�[�^�̒ǉ����I��������A��x�̂ݎ��s. ��������s����ƒl�����������Ȃ�̂Œ���. */
		Gravisbell::ErrorCode Normalize();


	public:
		/** �f�[�^�t�H�[�}�b�g�����擾���� */
		U32 GetDataFormatCount()const;

		/** �f�[�^�t�H�[�}�b�g��S�폜���� */
		Gravisbell::ErrorCode ClearDataFormat();

		/** �f�[�^�t�H�[�}�b�g��ǉ����� */
		Gravisbell::ErrorCode AddDataFormat(Format::CItem_base* pDataFormat);


	public:
		/** �o�C�i���f�[�^��ǂݍ���.
			@param	i_lpBuf		�o�C�i���擪�A�h���X.
			@param	i_byteCount	�Ǎ��\�ȃo�C�g��.
			@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
		S32 LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount);
	};


}	// Binary
}	// DataFormat
}	// Gravisbell


#endif	// __GRAVISBELL_DATAFORMAT_BINARY_DATAFORMAT_CLASS_H__