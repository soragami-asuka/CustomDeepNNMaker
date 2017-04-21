//====================================
// �f�[�^�t�H�[�}�b�g��`�̖{�̏��
//====================================
#include"stdafx.h"

#include"DataFormatClass.h"


namespace Gravisbell {
namespace DataFormat {
namespace Binary {

	/** �R���X�g���N�^ */
	CDataFormat::CDataFormat()
	:	CDataFormat	(L"", L"", false)
	{
	}
	/** �R���X�g���N�^ */
	CDataFormat::CDataFormat(const wchar_t i_szName[], const wchar_t i_szText[], bool onReverseByteOrder)
	:	name				(i_szName)
	,	text				(i_szText)
	,	onReverseByteOrder	(onReverseByteOrder)
	{
	}
	/** �f�X�g���N�^ */
	CDataFormat::~CDataFormat()
	{
		this->ClearDataFormat();

		// �f�[�^��S�폜
		for(auto& dataInfo : this->lpData)
		{
			for(U32 i=0; i<dataInfo.second.lpData.size(); i++)
				delete[] dataInfo.second.lpData[i];
		}
		this->lpData.clear();
	}



	/** ���O�̎擾 */
	const wchar_t* CDataFormat::GetName()const
	{
		return name.c_str();
	}
	/** �������̎擾 */
	const wchar_t* CDataFormat::GetText()const
	{
		return text.c_str();
	}

	/** X�����̗v�f�����擾 */
	U32 CDataFormat::GetBufferCountX(const wchar_t i_szCategory[])const
	{
		auto pDataInfo = GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return 0;

		return this->GetVariableValue(pDataInfo->dataStruct.m_x.c_str());
	}
	/** Y�����̗v�f�����擾 */
	U32 CDataFormat::GetBufferCountY(const wchar_t i_szCategory[])const
	{
		auto pDataInfo = GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return 0;

		return this->GetVariableValue(pDataInfo->dataStruct.m_y.c_str());
	}
	/** Z�����̗v�f�����擾 */
	U32 CDataFormat::GetBufferCountZ(const wchar_t i_szCategory[])const
	{
		auto pDataInfo = GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return 0;

		return this->GetVariableValue(pDataInfo->dataStruct.m_z.c_str());
	}
	/** CH�����̗v�f�����擾 */
	U32 CDataFormat::GetBufferCountCH(const wchar_t i_szCategory[])const
	{
		auto pDataInfo = GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return 0;

		return this->GetVariableValue(pDataInfo->dataStruct.m_ch.c_str());
	}

	/** �f�[�^�\�����擾 */
	IODataStruct CDataFormat::GetDataStruct(const wchar_t i_szCategory[])const
	{
		return IODataStruct(this->GetBufferCountCH(i_szCategory), this->GetBufferCountX(i_szCategory), this->GetBufferCountY(i_szCategory), this->GetBufferCountZ(i_szCategory));
	}

	/** �f�[�^�����擾 */
	const DataInfo* CDataFormat::GetDataInfo(const wchar_t i_szCategory[])const
	{
		auto it = this->lpData.find(i_szCategory);
		if(it == this->lpData.end())
			return NULL;

		return &it->second;
	}
	/** �f�[�^�����擾 */
	DataInfo* CDataFormat::GetDataInfo(const wchar_t i_szCategory[])
	{
		auto it = this->lpData.find(i_szCategory);
		if(it == this->lpData.end())
			return NULL;

		return &it->second;
	}

	/** �f�[�^����ǉ����� */
	Gravisbell::ErrorCode CDataFormat::AddDataInfo(const wchar_t i_szCategory[], const wchar_t i_x[], const wchar_t i_y[], const wchar_t i_z[], const wchar_t i_ch[], F32 i_false, F32 i_true)
	{
		if(this->lpData.count(i_szCategory) > 0)
			return ErrorCode::ERROR_CODE_COMMON_ADD_ALREADY_SAMEID;

		this->lpData[i_szCategory].dataStruct.m_x     = i_x;
		this->lpData[i_szCategory].dataStruct.m_y     = i_y;
		this->lpData[i_szCategory].dataStruct.m_z     = i_z;
		this->lpData[i_szCategory].dataStruct.m_ch    = i_ch;
		this->lpData[i_szCategory].dataStruct.m_false = i_false;
		this->lpData[i_szCategory].dataStruct.m_true  = i_true;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �f�[�^���ɒl���������� */
	Gravisbell::ErrorCode CDataFormat::SetDataValue(const wchar_t i_szCategory[], U32 i_no, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, F32 value)
	{
		DataInfo* pDataInfo = this->GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		if(pDataInfo->lpData.size() <= i_no)
		{
			// �o�b�t�@�T�C�Y���l��
			U32 bufferSize = this->GetDataStruct(i_szCategory).GetDataCount();
			if(bufferSize <= 0)
				return ErrorCode::ERROR_CODE_COMMON_ALLOCATION_MEMORY;

			// �f�[�^��������Ȃ����߃��������m��
			U32 pos = pDataInfo->lpData.size();

			// ���T�C�Y
			pDataInfo->lpData.resize(i_no+1);

			// �������m��
			for(; pos<pDataInfo->lpData.size(); pos++)
			{
				pDataInfo->lpData[pos] = new F32[bufferSize];

				// ���߂�
				for(U32 i=0; i<bufferSize; i++)
					pDataInfo->lpData[pos][i] = pDataInfo->dataStruct.m_false;
			}
		}

		// �l��ݒ肷��
		U32 size_x  = this->GetVariableValue(pDataInfo->dataStruct.m_x.c_str());
		if(i_x >= size_x)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
		U32 size_y  = this->GetVariableValue(pDataInfo->dataStruct.m_y.c_str());
		if(i_y >= size_y)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
		U32 size_z  = this->GetVariableValue(pDataInfo->dataStruct.m_z.c_str());
		if(i_z >= size_z)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
		U32 size_ch = this->GetVariableValue(pDataInfo->dataStruct.m_ch.c_str());
		if(i_ch >= size_ch)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		U32 pos = 
			(size_y * size_x * size_ch) * i_z +
			(         size_x * size_ch) * i_y +
			(                  size_ch) * i_x +
			i_ch;

		pDataInfo->lpData[i_no][pos] = value;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �f�[�^���ɒl����������.
		0.0=false.
		1.0=true.
		�Ƃ��Ēl���i�[���� */
	Gravisbell::ErrorCode CDataFormat::SetDataValueNormalize(const wchar_t i_szCategory[], U32 i_No, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, F32 value)
	{
		DataInfo* pDataInfo = this->GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		F32 tmpValue = (pDataInfo->dataStruct.m_true - pDataInfo->dataStruct.m_false) * value + pDataInfo->dataStruct.m_false;
		tmpValue = min(pDataInfo->dataStruct.m_true, max(pDataInfo->dataStruct.m_false, tmpValue));

		return this->SetDataValue(i_szCategory, i_No, i_x, i_y, i_z, i_ch, tmpValue);
	}
	/** �f�[�^���ɒl���������� */
	Gravisbell::ErrorCode CDataFormat::SetDataValue(const wchar_t i_szCategory[], U32 i_No, U32 i_x, U32 i_y, U32 i_z, U32 i_ch, bool value)
	{
		DataInfo* pDataInfo = this->GetDataInfo(i_szCategory);
		if(pDataInfo)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		return this->SetDataValue(i_szCategory, i_No, i_x, i_y, i_z, i_ch, value ? pDataInfo->dataStruct.m_true : pDataInfo->dataStruct.m_false);
	}


	/** ID�w��ŕϐ��̒l���擾����.(���l����t��) */
	S32 CDataFormat::GetVariableValue(const wchar_t i_szID[])const
	{
		if(isdigit(i_szID[0]))
		{
			// �ϐ��ł͂Ȃ�
			return ConvertString2Int(i_szID);
		}
		else
		{
			// �ϐ�
			auto it = this->lpVariable.find(i_szID);
			if(it == this->lpVariable.end())
				return 0;
			return it->second;
		}

		return 0;
	}
	/** ID�w��ŕϐ��̒l���擾����.(���l����t��.)(float�^�Ƃ��Ēl��Ԃ�) */
	F32 CDataFormat::GetVariableValueAsFloat(const wchar_t i_szID[])const
	{
		if(isdigit(i_szID[0]))
		{
			// �ϐ��ł͂Ȃ�
			return ConvertString2Float(i_szID);
		}
		else
		{
			// �ϐ�
			auto it = this->lpVariable.find(i_szID);
			if(it == this->lpVariable.end())
				return 0.0f;
			return (F32)it->second;
		}

		return 0;
	}
	/** ID�w��ŕϐ��ɒl��ݒ肷��.(���l����t��) */
	void CDataFormat::SetVariableValue(const wchar_t i_szID[], S32 value)
	{
		this->lpVariable[i_szID] = value;
	}
	/** �������l�ɕϊ�.�i������t�� */
	S32 ConvertString2Int(const std::wstring& buf)
	{
		if(buf.size() >= 2 && buf.c_str()[0] == L'0')	// ���̒l
		{
			// �i��������s��
			if(buf.c_str()[1] == L'x')
			{
				// 16�i��
				return wcstol(&buf.c_str()[2], NULL, 16);
			}
			else
			{
				// 8�i��
				return wcstol(&buf.c_str()[1], NULL, 8);
			}
		}
		if(buf.size() >= 3 && buf.c_str()[0] == L'-' && buf.c_str()[1] == L'0')	// ���̒l
		{
			// �i��������s��
			if(buf.c_str()[2] == L'x')
			{
				// 16�i��
				return -wcstol(&buf.c_str()[3], NULL, 16);
			}
			else
			{
				// 8�i��
				return -wcstol(&buf.c_str()[2], NULL, 8);
			}
		}
		else
		{
			return wcstol(buf.c_str(), NULL, 10);
		}
	}
	/** �������l�ɕϊ�.�i������t�� */
	U32 ConvertString2UInt(const std::wstring& buf)
	{
		return (U32)ConvertString2Int(buf);
	}
	/** �������l�ɕϊ�.�i������t�� */
	F32 ConvertString2Float(const std::wstring& buf)
	{
		if(buf.size() >= 2 && buf.c_str()[0] == L'0' && buf.c_str()[1] != L'.')	// ���̒l
		{
			// �i��������s��
			if(buf.c_str()[1] == L'x')
			{
				// 16�i��
				return (F32)wcstol(&buf.c_str()[2], NULL, 16);
			}
			else
			{
				// 8�i��
				return (F32)wcstol(&buf.c_str()[1], NULL, 8);
			}
		}
		if(buf.size() >= 3 && buf.c_str()[0] == L'-' && buf.c_str()[1] == L'0' && buf.c_str()[2] != L'.')	// ���̒l
		{
			// �i��������s��
			if(buf.c_str()[2] == L'x')
			{
				// 16�i��
				return (F32)-wcstol(&buf.c_str()[3], NULL, 16);
			}
			else
			{
				// 8�i��
				return (F32)-wcstol(&buf.c_str()[2], NULL, 8);
			}
		}
		else
		{
			// 10�i���̏ꍇ��float�^
			return (F32)wcstof(buf.c_str(), NULL);
		}
	}

	/** �J�e�S���[�����擾���� */
	U32 CDataFormat::GetCategoryCount()const
	{
		return this->lpData.size();
	}
	/** �J�e�S���[����ԍ��w��Ŏ擾���� */
	const wchar_t* CDataFormat::GetCategoryNameByNum(U32 categoryNo)const
	{
		if(categoryNo >= this->lpData.size())
			return NULL;

		auto it = this->lpData.begin();
		for(U32 no=0; no<categoryNo; no++)
			it++;
		return it->first.c_str();
	}

	/** Byte-Order�̔��]�t���O���擾���� */
	bool CDataFormat::GetOnReverseByteOrder()const
	{
		return this->onReverseByteOrder;
	}


	/** �f�[�^�����擾���� */
	U32 CDataFormat::GetDataCount()const
	{
		if(this->lpData.empty())
			return 0;

		return this->lpData.begin()->second.lpData.size();;
	}

	/** �f�[�^���擾���� */
	const F32* CDataFormat::GetDataByNum(U32 i_dataNo, const wchar_t i_szCategory[])const
	{
		auto pDataInfo = this->GetDataInfo(i_szCategory);
		if(pDataInfo == NULL)
			return NULL;

		if(pDataInfo->lpData.size() >= i_dataNo)
			return NULL;

		return pDataInfo->lpData[i_dataNo];
	}

	/** ���K������.
		�f�[�^�̒ǉ����I��������A��x�̂ݎ��s. ��������s����ƒl�����������Ȃ�̂Œ���. */
	Gravisbell::ErrorCode CDataFormat::Normalize()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �f�[�^�t�H�[�}�b�g�����擾���� */
	U32 CDataFormat::GetDataFormatCount()const
	{
		return this->lpDataFormat.size();
	}

	/** �f�[�^�t�H�[�}�b�g��S�폜���� */
	Gravisbell::ErrorCode CDataFormat::ClearDataFormat()
	{
		for(auto it : this->lpDataFormat)
		{
			delete it;
		}
		this->lpDataFormat.clear();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �f�[�^�t�H�[�}�b�g��ǉ����� */
	Gravisbell::ErrorCode CDataFormat::AddDataFormat(Format::CItem_base* pDataFormat)
	{
		this->lpDataFormat.push_back(pDataFormat);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�C�i���f�[�^��ǂݍ���.
		@param	i_lpBuf		�o�C�i���擪�A�h���X.
		@param	i_byteCount	�Ǎ��\�ȃo�C�g��.
		@return	���ۂɓǂݍ��񂾃o�C�g��. ���s�����ꍇ�͕��̒l */
	S32 CDataFormat::LoadBinary(const BYTE* i_lpBuf, U32 i_byteCount)
	{
		U32 bufPos = 0;

		for(auto pItem : this->lpDataFormat)
		{
			if(bufPos >= i_byteCount)
				return -1;

			S32 useBufNum = pItem->LoadBinary(&i_lpBuf[bufPos], i_byteCount-bufPos);
			if(useBufNum < 0)
				return useBufNum;

			bufPos += useBufNum;
		}

		return (S32)bufPos;
	}


}	// Binary
}	// DataFormat
}	// Gravisbell

